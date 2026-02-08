#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::kernels::utils;
use crate::tensor::TensorView;

/// AVX2-optimized dynamic quantize linear
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dynamic_quantize_linear_avx2<'a, 'b>(
    x: &TensorView<'b, f32>,
    out_y: &'a mut Vec<f32>,
    out_scale: &'a mut Vec<f32>,
    out_zp: &'a mut Vec<f32>,
) -> (
    TensorView<'a, f32>,
    TensorView<'a, f32>,
    TensorView<'a, f32>,
) {
    unsafe {
        let len = x.data.len();
        if len == 0 {
            return (
                TensorView::from_owned(vec![], x.shape.to_vec()),
                TensorView::from_owned(vec![1.0], vec![1]),
                TensorView::from_owned(vec![0.0], vec![1]),
            );
        }

        // SIMD min/max finding
        let mut min_vec = _mm256_set1_ps(f32::MAX);
        let mut max_vec = _mm256_set1_ps(f32::MIN);
        let mut i = 0;
        let simd_end = (len / 8) * 8;
        let ptr = x.data.as_ptr();

        while i < simd_end {
            let v = _mm256_loadu_ps(ptr.add(i));
            min_vec = _mm256_min_ps(min_vec, v);
            max_vec = _mm256_max_ps(max_vec, v);
            i += 8;
        }

        // Horizontal min/max
        let mut min_val = hmin_ps(min_vec);
        let mut max_val = hmax_ps(max_vec);

        for j in simd_end..len {
            let v = *ptr.add(j);
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        let adjusted_max = max_val.max(0.0);
        let adjusted_min = min_val.min(0.0);
        let range = (adjusted_max - adjusted_min).max(1e-5);
        let scale = range / 255.0;
        let zp = (-adjusted_min / scale).round().clamp(0.0, 255.0);
        let inv_scale = 1.0 / scale;

        out_scale.clear();
        out_scale.push(scale);
        out_zp.clear();
        out_zp.push(zp);

        utils::ensure_capacity(out_y, len);

        // SIMD quantization
        let inv_scale_vec = _mm256_set1_ps(inv_scale);
        let zp_vec = _mm256_set1_ps(zp);
        let zero_vec = _mm256_setzero_ps();
        let max_255 = _mm256_set1_ps(255.0);
        let out_ptr = out_y.as_mut_ptr();

        i = 0;
        while i + 8 <= len {
            let v = _mm256_loadu_ps(ptr.add(i));
            let scaled = _mm256_fmadd_ps(v, inv_scale_vec, zp_vec);
            let rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let clamped = _mm256_min_ps(_mm256_max_ps(rounded, zero_vec), max_255);
            _mm256_storeu_ps(out_ptr.add(i), clamped);
            i += 8;
        }

        for j in i..len {
            *out_ptr.add(j) = (*ptr.add(j) * inv_scale + zp).round().clamp(0.0, 255.0);
        }

        (
            TensorView::from_slice(out_y, x.shape.to_vec()),
            TensorView::from_slice(out_scale, vec![1]),
            TensorView::from_slice(out_zp, vec![1]),
        )
    }
}

/// AVX2-optimized u8 matrix multiply using VPMADDUBSW integer dot product.
/// Uses true int8 SIMD: processes 32 u8 multiplies per instruction (4x over f32 FMA path).
///
/// Math: out[i][j] = sum_k (a[i][k] - zp_a) * (b[k][j] - zp_b)
///
/// Rewrite for VPMADDUBSW (u8 × i8 → i16):
///   b_u8 as i8 = b_u8 - 128 (reinterpret cast, wraps correctly)
///   So: a_u8 * b_i8_reinterp = a_u8 * (b_u8 - 128)
///   Then: (a - zp_a)(b - zp_b) = a*b_i8 + a*(128 - zp_b) - zp_a*(b_col_sum - K*zp_b)
///   where b_col_sum = sum_k b[k][j] (original u8 values)
///
/// We pre-transpose B to [N, K_padded] and store as i8 (reinterpreted from u8 by XOR 0x80).
/// Per-row correction: row_sum_a * (128 - zp_b) - zp_a * col_sum_b_adj
/// where col_sum_b_adj = col_sum_b_u8 - K * 128  (sum of the reinterpreted i8 values)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn mat_mul_integer_u8_avx2<'a, 'b, 'c>(
    a: &TensorView<'b, u8>,
    b: &TensorView<'c, u8>,
    a_zero_point: Option<&TensorView<'b, u8>>,
    b_zero_point: Option<&TensorView<'c, u8>>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    unsafe {
        let zp_a = a_zero_point.map(|z| z.data[0]).unwrap_or(0u8) as i32;
        let zp_b = b_zero_point.map(|z| z.data[0]).unwrap_or(0u8) as i32;

        let a_dims = a.shape.len();
        let b_dims = b.shape.len();
        let m = a.shape[a_dims - 2];
        let k = a.shape[a_dims - 1];
        let n = b.shape[b_dims - 1];

        let batch_a: usize = a.shape[..a_dims - 2].iter().product();
        let batch_b: usize = b.shape[..b_dims - 2].iter().product();
        let final_batch = batch_a.max(batch_b);
        let output_len = final_batch * m * n;

        utils::ensure_capacity(out, output_len);
        out.resize(output_len, 0.0);

        let stride_a = m * k;
        let stride_b = k * n;
        let stride_out = m * n;

        // Pad K to multiple of 16 for SIMD (VPMADDWD processes 16 pairs per instruction)
        let k_padded = (k + 15) & !15;

        // Pre-transpose B to [N, K_padded] with u8 XOR 0x80 → reinterpret as i8
        // Also compute col_sums of reinterpreted i8 values for zero-point correction
        let mut b_t = vec![0u8; n * k_padded]; // stored as u8, interpreted as i8 by sign-extend

        // Correction constant per column:
        // (a - zp_a)(b - zp_b) = a*(b^0x80 - 128 + 128 - zp_b) - zp_a*(b - zp_b)
        //                      = a*(b_reinterp) + a*(128 - zp_b) - zp_a*col_sum_b_orig + zp_a*zp_b*K
        // where b_reinterp = b^0x80 interpreted as i8 = b_u8 - 128
        // So VPMADDUBSW(a_u8, b_reinterp_i8) gives us: sum(a * (b - 128))
        // We need: sum((a - zp_a)(b - zp_b)) = sum(a*(b-128)) + sum(a)*(128 - zp_b) - zp_a*sum(b-zp_b)
        //        = dot_result + row_sum_a * (128 - zp_b) - zp_a * (col_sum_b_u8 - K * zp_b)

        // col_sum_b_u8[j] = sum over k of b_u8[k][j]
        let mut col_sums_b_u8 = vec![0i32; n];

        for b_i in 0..final_batch {
            let a_offset = if batch_a == 1 { 0 } else { b_i * stride_a };
            let b_offset = if batch_b == 1 { 0 } else { b_i * stride_b };
            let out_offset = b_i * stride_out;

            let a_slice = &a.data[a_offset..a_offset + stride_a];
            let b_slice = &b.data[b_offset..b_offset + stride_b];
            let out_slice = &mut out[out_offset..out_offset + stride_out];

            // Transpose B and XOR with 0x80: B[kk][jj] -> B_T[jj][kk] = B[kk][jj] ^ 0x80
            if b_i == 0 || batch_b > 1 {
                for jj in 0..n {
                    let mut csum: i32 = 0;
                    for kk in 0..k {
                        let b_val = b_slice[kk * n + jj];
                        b_t[jj * k_padded + kk] = b_val ^ 0x80; // reinterpret as i8
                        csum += b_val as i32;
                    }
                    // Zero-pad remainder (0x00 as i8 = 0, correct for zero-padding)
                    for kk in k..k_padded {
                        b_t[jj * k_padded + kk] = 0x80; // 0x80 ^ 0x80 = 0 as i8, but we XOR'd already
                        // Actually: padding should contribute 0 to the dot product.
                        // a_padded[kk] = 0 (we'll pad A with 0), so b_t value doesn't matter
                        // But A is padded with 0, so this is fine.
                        b_t[jj * k_padded + kk] = 0; // doesn't matter, A padding is 0
                    }
                    col_sums_b_u8[jj] = csum;
                }
            }

            // Pre-compute per-column correction: zp_a * (col_sum_b_u8[j] - K * zp_b)
            let k_zp_b = k as i32 * zp_b;
            let corr_128_minus_zpb = 128 - zp_b; // (128 - zp_b) factor

            // Prepare A with zero-padding to k_padded if needed
            // We'll copy A row by row with padding during the loop

            for i in 0..m {
                let a_row_start = i * k;
                let a_row = &a_slice[a_row_start..a_row_start + k];

                // Compute row_sum_a for zero-point correction
                let mut row_sum_a: i32 = 0;
                for &av in a_row {
                    row_sum_a += av as i32;
                }

                // Prepare padded A row (stack for small K, heap for large)
                let mut a_padded_heap;
                let mut a_padded_stack = [0u8; 2080]; // 2048 + 32
                let a_padded: &mut [u8] = if k_padded <= 2080 {
                    a_padded_stack[..k_padded].fill(0);
                    &mut a_padded_stack[..k_padded]
                } else {
                    a_padded_heap = vec![0u8; k_padded];
                    &mut a_padded_heap
                };
                a_padded[..k].copy_from_slice(a_row);

                let a_ptr = a_padded.as_ptr();

                // Per-row correction part: row_sum_a * (128 - zp_b)
                let row_corr = row_sum_a * corr_128_minus_zpb;

                let _zero = _mm256_setzero_si256();

                let mut j = 0;
                while j + 4 <= n {
                    let bt_ptr0 = b_t.as_ptr().add(j * k_padded);
                    let bt_ptr1 = b_t.as_ptr().add((j + 1) * k_padded);
                    let bt_ptr2 = b_t.as_ptr().add((j + 2) * k_padded);
                    let bt_ptr3 = b_t.as_ptr().add((j + 3) * k_padded);

                    let mut iacc0 = _mm256_setzero_si256();
                    let mut iacc1 = _mm256_setzero_si256();
                    let mut iacc2 = _mm256_setzero_si256();
                    let mut iacc3 = _mm256_setzero_si256();

                    let mut kk = 0;
                    while kk + 16 <= k_padded {
                        // Load 16 u8 values from A, zero-extend to 16 x i16
                        let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                        let va_lo = _mm256_cvtepu8_epi16(a_128);

                        // Load 16 bytes from B_T, sign-extend to 16 x i16
                        // b_t contains u8 values that represent i8 (via XOR 0x80)
                        let b0_128 = _mm_loadu_si128(bt_ptr0.add(kk) as *const __m128i);
                        let b1_128 = _mm_loadu_si128(bt_ptr1.add(kk) as *const __m128i);
                        let b2_128 = _mm_loadu_si128(bt_ptr2.add(kk) as *const __m128i);
                        let b3_128 = _mm_loadu_si128(bt_ptr3.add(kk) as *const __m128i);

                        let vb0_lo = _mm256_cvtepi8_epi16(b0_128);
                        let vb1_lo = _mm256_cvtepi8_epi16(b1_128);
                        let vb2_lo = _mm256_cvtepi8_epi16(b2_128);
                        let vb3_lo = _mm256_cvtepi8_epi16(b3_128);

                        // VPMADDWD: i16 * i16 → i32 (adjacent pairs summed, NO saturation)
                        // Each produces 8 x i32 from 16 x i16 pairs
                        iacc0 = _mm256_add_epi32(iacc0, _mm256_madd_epi16(va_lo, vb0_lo));
                        iacc1 = _mm256_add_epi32(iacc1, _mm256_madd_epi16(va_lo, vb1_lo));
                        iacc2 = _mm256_add_epi32(iacc2, _mm256_madd_epi16(va_lo, vb2_lo));
                        iacc3 = _mm256_add_epi32(iacc3, _mm256_madd_epi16(va_lo, vb3_lo));

                        kk += 16;
                    }

                    // dot = sum(a_u8 * (b_u8 - 128))
                    let dot0 = hsum_epi32(iacc0);
                    let dot1 = hsum_epi32(iacc1);
                    let dot2 = hsum_epi32(iacc2);
                    let dot3 = hsum_epi32(iacc3);

                    // Full formula:
                    // (a - zp_a)(b - zp_b) = dot + row_sum_a*(128 - zp_b) - zp_a*(col_sum_b_u8 - K*zp_b)
                    let col_corr0 = zp_a * (col_sums_b_u8[j] - k_zp_b);
                    let col_corr1 = zp_a * (col_sums_b_u8[j + 1] - k_zp_b);
                    let col_corr2 = zp_a * (col_sums_b_u8[j + 2] - k_zp_b);
                    let col_corr3 = zp_a * (col_sums_b_u8[j + 3] - k_zp_b);

                    let mut f0 = (dot0 + row_corr - col_corr0) as f32;
                    let mut f1 = (dot1 + row_corr - col_corr1) as f32;
                    let mut f2 = (dot2 + row_corr - col_corr2) as f32;
                    let mut f3 = (dot3 + row_corr - col_corr3) as f32;

                    // Apply scale
                    if let Some(scale_data) = scale {
                        if scale_data.data.len() == 1 {
                            let sv = scale_data.data[0];
                            f0 *= sv; f1 *= sv; f2 *= sv; f3 *= sv;
                        } else {
                            f0 *= scale_data.data[j];
                            f1 *= scale_data.data[j + 1];
                            f2 *= scale_data.data[j + 2];
                            f3 *= scale_data.data[j + 3];
                        }
                    }
                    // Apply bias
                    if let Some(bias_data) = bias {
                        f0 += bias_data.data[j];
                        f1 += bias_data.data[j + 1];
                        f2 += bias_data.data[j + 2];
                        f3 += bias_data.data[j + 3];
                    }
                    // Apply ReLU
                    if apply_relu {
                        f0 = f0.max(0.0);
                        f1 = f1.max(0.0);
                        f2 = f2.max(0.0);
                        f3 = f3.max(0.0);
                    }

                    out_slice[i * n + j] = f0;
                    out_slice[i * n + j + 1] = f1;
                    out_slice[i * n + j + 2] = f2;
                    out_slice[i * n + j + 3] = f3;

                    j += 4;
                }

                // Handle remaining columns
                while j < n {
                    let bt_ptr = b_t.as_ptr().add(j * k_padded);
                    let mut iacc = _mm256_setzero_si256();
                    let mut kk = 0;
                    while kk + 16 <= k_padded {
                        let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                        let va_lo = _mm256_cvtepu8_epi16(a_128);
                        let b_128 = _mm_loadu_si128(bt_ptr.add(kk) as *const __m128i);
                        let vb_lo = _mm256_cvtepi8_epi16(b_128);
                        iacc = _mm256_add_epi32(iacc, _mm256_madd_epi16(va_lo, vb_lo));
                        kk += 16;
                    }
                    let dot = hsum_epi32(iacc);
                    let col_corr = zp_a * (col_sums_b_u8[j] - k_zp_b);
                    let mut sum = (dot + row_corr - col_corr) as f32;
                    if let Some(scale_data) = scale {
                        if scale_data.data.len() == 1 { sum *= scale_data.data[0]; }
                        else { sum *= scale_data.data[j]; }
                    }
                    if let Some(bias_data) = bias { sum += bias_data.data[j]; }
                    if apply_relu && sum < 0.0 { sum = 0.0; }
                    out_slice[i * n + j] = sum;
                    j += 1;
                }
            }
        }

        let mut output_shape = if batch_a >= batch_b {
            a.shape[..a_dims - 2].to_vec()
        } else {
            b.shape[..b_dims - 2].to_vec()
        };
        output_shape.push(m);
        output_shape.push(n);

        TensorView::from_slice(out, output_shape)
    }
}

/// Compute one row of the output matrix C = A_row × B_t (with zero-point corrections).
/// This is the core GEMM kernel, designed to be called from parallel or sequential paths.
/// `a_row`: u8 slice [K], `out_row`: f32 slice [N]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_row_avx2(
    a_row: &[u8],
    b_t_ptr: *const u8,     // Pre-transposed [N, K_padded] with XOR 0x80
    k: usize,
    n: usize,
    k_padded: usize,
    k_aligned: bool,
    col_sums_ptr: *const i32,
    zp_a: i32,
    _zp_b: i32,
    k_zp_b: i32,
    corr_128_minus_zpb: i32,
    scale_data_ptr: Option<*const f32>,
    scale_len: usize,
    bias_data_ptr: Option<*const f32>,
    apply_relu: bool,
    out_row: &mut [f32],
) {
    unsafe {
        // Vectorized row_sum_a using _mm256_sad_epu8
        let mut row_sum_a: i32;
        let zero_vec = _mm256_setzero_si256();
        {
            let mut sad_acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 32 <= k {
                let va = _mm256_loadu_si256(a_row.as_ptr().add(kk) as *const __m256i);
                sad_acc = _mm256_add_epi64(sad_acc, _mm256_sad_epu8(va, zero_vec));
                kk += 32;
            }
            let hi = _mm256_extracti128_si256(sad_acc, 1);
            let lo = _mm256_castsi256_si128(sad_acc);
            let s128 = _mm_add_epi64(lo, hi);
            row_sum_a = (_mm_extract_epi64(s128, 0) + _mm_extract_epi64(s128, 1)) as i32;
            while kk < k {
                row_sum_a += a_row[kk] as i32;
                kk += 1;
            }
        }

        // Use A directly if aligned, otherwise pad
        let a_ptr: *const u8;
        let mut a_padded_stack = [0u8; 2080];
        if k_aligned {
            a_ptr = a_row.as_ptr();
        } else {
            a_padded_stack[..k].copy_from_slice(a_row);
            a_ptr = a_padded_stack.as_ptr();
        }

        let row_corr = row_sum_a * corr_128_minus_zpb;

        let mut j = 0;
        // 8-column unrolling
        while j + 8 <= n {
            let bt0 = b_t_ptr.add(j * k_padded);
            let bt1 = b_t_ptr.add((j + 1) * k_padded);
            let bt2 = b_t_ptr.add((j + 2) * k_padded);
            let bt3 = b_t_ptr.add((j + 3) * k_padded);
            let bt4 = b_t_ptr.add((j + 4) * k_padded);
            let bt5 = b_t_ptr.add((j + 5) * k_padded);
            let bt6 = b_t_ptr.add((j + 6) * k_padded);
            let bt7 = b_t_ptr.add((j + 7) * k_padded);

            let mut iacc0 = _mm256_setzero_si256();
            let mut iacc1 = _mm256_setzero_si256();
            let mut iacc2 = _mm256_setzero_si256();
            let mut iacc3 = _mm256_setzero_si256();
            let mut iacc4 = _mm256_setzero_si256();
            let mut iacc5 = _mm256_setzero_si256();
            let mut iacc6 = _mm256_setzero_si256();
            let mut iacc7 = _mm256_setzero_si256();

            let mut kk = 0;
            while kk + 16 <= k_padded {
                let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                let va = _mm256_cvtepu8_epi16(a_128);

                iacc0 = _mm256_add_epi32(iacc0, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt0.add(kk) as *const __m128i))));
                iacc1 = _mm256_add_epi32(iacc1, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt1.add(kk) as *const __m128i))));
                iacc2 = _mm256_add_epi32(iacc2, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt2.add(kk) as *const __m128i))));
                iacc3 = _mm256_add_epi32(iacc3, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt3.add(kk) as *const __m128i))));
                iacc4 = _mm256_add_epi32(iacc4, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt4.add(kk) as *const __m128i))));
                iacc5 = _mm256_add_epi32(iacc5, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt5.add(kk) as *const __m128i))));
                iacc6 = _mm256_add_epi32(iacc6, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt6.add(kk) as *const __m128i))));
                iacc7 = _mm256_add_epi32(iacc7, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt7.add(kk) as *const __m128i))));

                kk += 16;
            }

            // Vectorized horizontal sum using hadd
            let h01 = _mm256_hadd_epi32(iacc0, iacc1);
            let h23 = _mm256_hadd_epi32(iacc2, iacc3);
            let h0123 = _mm256_hadd_epi32(h01, h23);
            let h0123_hi = _mm256_extracti128_si256(h0123, 1);
            let h0123_lo = _mm256_castsi256_si128(h0123);
            let dots_0123 = _mm_add_epi32(h0123_lo, h0123_hi);

            let h45 = _mm256_hadd_epi32(iacc4, iacc5);
            let h67 = _mm256_hadd_epi32(iacc6, iacc7);
            let h4567 = _mm256_hadd_epi32(h45, h67);
            let h4567_hi = _mm256_extracti128_si256(h4567, 1);
            let h4567_lo = _mm256_castsi256_si128(h4567);
            let dots_4567 = _mm_add_epi32(h4567_lo, h4567_hi);

            let d0 = _mm_extract_epi32(dots_0123, 0);
            let d1 = _mm_extract_epi32(dots_0123, 1);
            let d2 = _mm_extract_epi32(dots_0123, 2);
            let d3 = _mm_extract_epi32(dots_0123, 3);
            let d4 = _mm_extract_epi32(dots_4567, 0);
            let d5 = _mm_extract_epi32(dots_4567, 1);
            let d6 = _mm_extract_epi32(dots_4567, 2);
            let d7 = _mm_extract_epi32(dots_4567, 3);

            let mut f0 = (d0 + row_corr - zp_a * (*col_sums_ptr.add(j) - k_zp_b)) as f32;
            let mut f1 = (d1 + row_corr - zp_a * (*col_sums_ptr.add(j+1) - k_zp_b)) as f32;
            let mut f2 = (d2 + row_corr - zp_a * (*col_sums_ptr.add(j+2) - k_zp_b)) as f32;
            let mut f3 = (d3 + row_corr - zp_a * (*col_sums_ptr.add(j+3) - k_zp_b)) as f32;
            let mut f4 = (d4 + row_corr - zp_a * (*col_sums_ptr.add(j+4) - k_zp_b)) as f32;
            let mut f5 = (d5 + row_corr - zp_a * (*col_sums_ptr.add(j+5) - k_zp_b)) as f32;
            let mut f6 = (d6 + row_corr - zp_a * (*col_sums_ptr.add(j+6) - k_zp_b)) as f32;
            let mut f7 = (d7 + row_corr - zp_a * (*col_sums_ptr.add(j+7) - k_zp_b)) as f32;

            if let Some(sp) = scale_data_ptr {
                if scale_len == 1 {
                    let sv = *sp;
                    f0 *= sv; f1 *= sv; f2 *= sv; f3 *= sv;
                    f4 *= sv; f5 *= sv; f6 *= sv; f7 *= sv;
                } else {
                    f0 *= *sp.add(j); f1 *= *sp.add(j+1);
                    f2 *= *sp.add(j+2); f3 *= *sp.add(j+3);
                    f4 *= *sp.add(j+4); f5 *= *sp.add(j+5);
                    f6 *= *sp.add(j+6); f7 *= *sp.add(j+7);
                }
            }
            if let Some(bp) = bias_data_ptr {
                f0 += *bp.add(j); f1 += *bp.add(j+1);
                f2 += *bp.add(j+2); f3 += *bp.add(j+3);
                f4 += *bp.add(j+4); f5 += *bp.add(j+5);
                f6 += *bp.add(j+6); f7 += *bp.add(j+7);
            }
            if apply_relu {
                f0 = f0.max(0.0); f1 = f1.max(0.0); f2 = f2.max(0.0); f3 = f3.max(0.0);
                f4 = f4.max(0.0); f5 = f5.max(0.0); f6 = f6.max(0.0); f7 = f7.max(0.0);
            }

            out_row[j] = f0; out_row[j+1] = f1; out_row[j+2] = f2; out_row[j+3] = f3;
            out_row[j+4] = f4; out_row[j+5] = f5; out_row[j+6] = f6; out_row[j+7] = f7;

            j += 8;
        }

        // 4-column remainder
        while j + 4 <= n {
            let bt0 = b_t_ptr.add(j * k_padded);
            let bt1 = b_t_ptr.add((j + 1) * k_padded);
            let bt2 = b_t_ptr.add((j + 2) * k_padded);
            let bt3 = b_t_ptr.add((j + 3) * k_padded);

            let mut iacc0 = _mm256_setzero_si256();
            let mut iacc1 = _mm256_setzero_si256();
            let mut iacc2 = _mm256_setzero_si256();
            let mut iacc3 = _mm256_setzero_si256();

            let mut kk = 0;
            while kk + 16 <= k_padded {
                let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                let va = _mm256_cvtepu8_epi16(a_128);

                iacc0 = _mm256_add_epi32(iacc0, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt0.add(kk) as *const __m128i))));
                iacc1 = _mm256_add_epi32(iacc1, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt1.add(kk) as *const __m128i))));
                iacc2 = _mm256_add_epi32(iacc2, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt2.add(kk) as *const __m128i))));
                iacc3 = _mm256_add_epi32(iacc3, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt3.add(kk) as *const __m128i))));

                kk += 16;
            }

            let h01 = _mm256_hadd_epi32(iacc0, iacc1);
            let h23 = _mm256_hadd_epi32(iacc2, iacc3);
            let h0123 = _mm256_hadd_epi32(h01, h23);
            let hi = _mm256_extracti128_si256(h0123, 1);
            let lo = _mm256_castsi256_si128(h0123);
            let dots = _mm_add_epi32(lo, hi);

            let d0 = _mm_extract_epi32(dots, 0);
            let d1 = _mm_extract_epi32(dots, 1);
            let d2 = _mm_extract_epi32(dots, 2);
            let d3 = _mm_extract_epi32(dots, 3);

            let mut f0 = (d0 + row_corr - zp_a * (*col_sums_ptr.add(j) - k_zp_b)) as f32;
            let mut f1 = (d1 + row_corr - zp_a * (*col_sums_ptr.add(j+1) - k_zp_b)) as f32;
            let mut f2 = (d2 + row_corr - zp_a * (*col_sums_ptr.add(j+2) - k_zp_b)) as f32;
            let mut f3 = (d3 + row_corr - zp_a * (*col_sums_ptr.add(j+3) - k_zp_b)) as f32;

            if let Some(sp) = scale_data_ptr {
                if scale_len == 1 {
                    let sv = *sp;
                    f0 *= sv; f1 *= sv; f2 *= sv; f3 *= sv;
                } else {
                    f0 *= *sp.add(j); f1 *= *sp.add(j+1);
                    f2 *= *sp.add(j+2); f3 *= *sp.add(j+3);
                }
            }
            if let Some(bp) = bias_data_ptr {
                f0 += *bp.add(j); f1 += *bp.add(j+1);
                f2 += *bp.add(j+2); f3 += *bp.add(j+3);
            }
            if apply_relu {
                f0 = f0.max(0.0); f1 = f1.max(0.0);
                f2 = f2.max(0.0); f3 = f3.max(0.0);
            }

            out_row[j] = f0; out_row[j+1] = f1;
            out_row[j+2] = f2; out_row[j+3] = f3;

            j += 4;
        }

        // Scalar remainder columns
        while j < n {
            let bt = b_t_ptr.add(j * k_padded);
            let mut iacc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 16 <= k_padded {
                let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                let va = _mm256_cvtepu8_epi16(a_128);
                let b_128 = _mm_loadu_si128(bt.add(kk) as *const __m128i);
                iacc = _mm256_add_epi32(iacc, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(b_128)));
                kk += 16;
            }
            let dot = hsum_epi32(iacc);
            let col_corr = zp_a * (*col_sums_ptr.add(j) - k_zp_b);
            let mut sum = (dot + row_corr - col_corr) as f32;
            if let Some(sp) = scale_data_ptr {
                if scale_len == 1 { sum *= *sp; }
                else { sum *= *sp.add(j); }
            }
            if let Some(bp) = bias_data_ptr { sum += *bp.add(j); }
            if apply_relu && sum < 0.0 { sum = 0.0; }
            out_row[j] = sum;
            j += 1;
        }
    }
}

/// Horizontal sum of 8 x i32 in __m256i -> single i32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32(v: __m256i) -> i32 {
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 1);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

/// Horizontal min of 8 f32s in __m256
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmin_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let m128 = _mm_min_ps(lo, hi);
    let m64 = _mm_min_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_min_ss(m64, _mm_shuffle_ps(m64, m64, 1));
    _mm_cvtss_f32(m32)
}

/// Horizontal max of 8 f32s in __m256
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(lo, hi);
    let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
    _mm_cvtss_f32(m32)
}
