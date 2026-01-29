use crate::kernels::utils;
use crate::tensor::TensorView;
use std::borrow::Cow;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use core::arch::asm;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vdotq_u32_custom(mut acc: uint32x4_t, a: uint8x16_t, b: uint8x16_t) -> uint32x4_t { unsafe {
    asm!(
        "udot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nostack)
    );
    acc
}}

pub fn dynamic_quantize_linear<'a, 'b>(
    x: &TensorView<'b, f32>,
    out_y: &'a mut Vec<f32>,
    out_scale: &'a mut Vec<f32>,
    out_zp: &'a mut Vec<f32>,
) -> (
    TensorView<'a, f32>,
    TensorView<'a, f32>,
    TensorView<'a, f32>,
) {
    // Legacy fake quantization fallback
    let len = x.data.len();
    if len == 0 {
        return (
            TensorView::from_owned(vec![], x.shape.to_vec()),
            TensorView::from_owned(vec![1.0], vec![1]),
            TensorView::from_owned(vec![0.0], vec![1]),
        );
    }
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in x.data.iter() {
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

    utils::ensure_capacity(out_scale, 1);
    out_scale[0] = scale;
    utils::ensure_capacity(out_zp, 1);
    out_zp[0] = zp;
    utils::ensure_capacity(out_y, len);

    let inv_scale = 1.0 / scale;
    for i in 0..len {
        out_y[i] = (x.data[i] * inv_scale + zp).round().clamp(0.0, 255.0);
    }
    (
        TensorView {
            data: Cow::Borrowed(out_y),
            shape: Cow::Owned(x.shape.to_vec()),
        },
        TensorView {
            data: Cow::Borrowed(out_scale),
            shape: Cow::Owned(vec![1]),
        },
        TensorView {
            data: Cow::Borrowed(out_zp),
            shape: Cow::Owned(vec![1]),
        },
    )
}

pub fn dynamic_quantize_linear_u8<'a, 'b>(
    x: &TensorView<'b, f32>,
    out_y_storage: &'a mut Vec<f32>,
    out_scale: &'a mut Vec<f32>,
    out_zp: &'a mut Vec<f32>,
) -> (TensorView<'a, u8>, TensorView<'a, f32>, TensorView<'a, u8>) {
    let len = x.data.len();
    // We store u8 data in a f32 vector by casting the pointer.
    // Ensure enough capacity: 1 f32 = 4 u8.
    let cap_bytes = out_y_storage.capacity() * 4;
    if cap_bytes < len {
        out_y_storage.reserve(len.div_ceil(4));
    }

    // IMPORTANT: This reuses the memory of a Vec<f32> to store u8.
    // The caller must be aware that `out_y_storage` now contains raw u8 bytes
    // effectively, even though it's typed as Vec<f32>.
    let ptr = out_y_storage.as_mut_ptr() as *mut u8;
    let out_u8 = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

    #[allow(unused)]
    let mut min_val = f32::MAX;
    #[allow(unused)]
    let mut max_val = f32::MIN;

    // Vectorized Min/Max
    let mut cur = 0;
    unsafe {
        let mut v_min = vdupq_n_f32(f32::MAX);
        let mut v_max = vdupq_n_f32(f32::MIN);

        while cur + 4 <= len {
            let v_data = vld1q_f32(x.data.as_ptr().add(cur));
            v_min = vminnmq_f32(v_min, v_data);
            v_max = vmaxnmq_f32(v_max, v_data);
            cur += 4;
        }

        // Reduce min
        min_val = vminvq_f32(v_min);
        max_val = vmaxvq_f32(v_max);
    }

    // Remainder Min/Max
    for j in cur..len {
        let v = x.data[j];
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

    utils::ensure_capacity(out_scale, 1);
    out_scale[0] = scale;

    utils::ensure_capacity(out_zp, 1);
    let ptr_zp = out_zp.as_mut_ptr() as *mut u8;
    let out_zp_u8 = unsafe { std::slice::from_raw_parts_mut(ptr_zp, 1) };
    out_zp_u8[0] = zp as u8;

    // Vectorized Quantization
    let mut cur = 0;
    unsafe {
        let v_inv_scale = vdupq_n_f32(inv_scale);
        let v_zp = vdupq_n_f32(zp);

        while cur + 16 <= len {
            // Process 16 float elements -> 16 u8 elements (1 vector)
            let v0 = vld1q_f32(x.data.as_ptr().add(cur));
            let v1 = vld1q_f32(x.data.as_ptr().add(cur + 4));
            let v2 = vld1q_f32(x.data.as_ptr().add(cur + 8));
            let v3 = vld1q_f32(x.data.as_ptr().add(cur + 12));

            // Mul + Add
            let r0 = vfmaq_f32(v_zp, v0, v_inv_scale);
            let r1 = vfmaq_f32(v_zp, v1, v_inv_scale);
            let r2 = vfmaq_f32(v_zp, v2, v_inv_scale);
            let r3 = vfmaq_f32(v_zp, v3, v_inv_scale);

            // Convert to u32 (Round to nearest)
            let i0 = vcvtaq_u32_f32(r0);
            let i1 = vcvtaq_u32_f32(r1);
            let i2 = vcvtaq_u32_f32(r2);
            let i3 = vcvtaq_u32_f32(r3);

            // Narrow u32 -> u16 (Saturate)
            let n0 = vqmovn_u32(i0); // 4x u16
            let n1 = vqmovn_u32(i1);
            let n2 = vqmovn_u32(i2);
            let n3 = vqmovn_u32(i3);

            // Combine n0/n1 -> 8x u16, n2/n3 -> 8x u16
            let nn0 = vcombine_u16(n0, n1);
            let nn1 = vcombine_u16(n2, n3);

            // Narrow u16 -> u8 (Saturate)
            let b0 = vqmovn_u16(nn0); // 8x u8
            let b1 = vqmovn_u16(nn1);

            // Combine -> 16x u8
            let res = vcombine_u8(b0, b1);

            vst1q_u8(out_u8.as_mut_ptr().add(cur), res);
            cur += 16;
        }
    }

    for j in cur..len {
        out_u8[j] = (x.data[j] * inv_scale + zp).round().clamp(0.0, 255.0) as u8;
    }

    (
        TensorView {
            data: Cow::Borrowed(out_u8),
            shape: Cow::Owned(x.shape.to_vec()),
        },
        TensorView::new(out_scale, &[1]),
        TensorView::new(out_zp_u8, &[1]),
    )
}

pub fn mat_mul_integer_u8<'a, 'b, 'c>(
    a: &TensorView<'b, u8>,
    b: &TensorView<'c, u8>,
    a_zero_point: Option<&TensorView<'b, u8>>,
    b_zero_point: Option<&TensorView<'c, u8>>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let zp_a = a_zero_point.map(|z| z.data[0] as i32).unwrap_or(0);
    let zp_b = b_zero_point.map(|z| z.data[0] as i32).unwrap_or(0);

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

    let constant_term = (k as f32) * (zp_a as f32) * (zp_b as f32);

    for b_i in 0..final_batch {
        let a_offset = if batch_a == 1 { 0 } else { b_i * stride_a };
        let b_offset = if batch_b == 1 { 0 } else { b_i * stride_b };
        let out_offset = b_i * stride_out;

        let a_slice = &a.data[a_offset..a_offset + stride_a];
        let b_slice = &b.data[b_offset..b_offset + stride_b];
        let out_slice = &mut out[out_offset..out_offset + stride_out];

        let mut row_sums_a = vec![0; m];
        if zp_b != 0 {
            for r in 0..m {
                let r_off = r * k;
                let mut s: i32 = 0;
                for c in 0..k {
                    s += a_slice[r_off + c] as i32;
                }
                row_sums_a[r] = s;
            }
        }

        let mut col_sums_b = vec![0; n];
        if zp_a != 0 {
            // Access B sequentially to be cache friendly
            // B is [k, n]
            for r in 0..k {
                let row_off = r * n;
                let mut c = 0;

                // Vectorized accumulation for col_sums
                unsafe {
                    while c + 16 <= n {
                        // Load 16 u8s
                        let v_b = vld1q_u8(b_slice.as_ptr().add(row_off + c));
                        // Expand to u16
                        let v_b_low = vget_low_u8(v_b);
                        let v_b_high = vget_high_u8(v_b);

                        let v_b_u16_0 = vmovl_u8(v_b_low);
                        let v_b_u16_1 = vmovl_u8(v_b_high);

                        // Expand to u32
                        let v_b_u32_0 = vmovl_u16(vget_low_u16(v_b_u16_0));
                        let v_b_u32_1 = vmovl_u16(vget_high_u16(v_b_u16_0));
                        let v_b_u32_2 = vmovl_u16(vget_low_u16(v_b_u16_1));
                        let v_b_u32_3 = vmovl_u16(vget_high_u16(v_b_u16_1));

                        // Load current sums (i32) - wait, col_sums_b is i32
                        // but we are initializing it. We can't overwrite if we are accumulating row by row.
                        // We need to Load, Add, Store.
                        let p_sum = col_sums_b.as_mut_ptr().add(c);
                        let v_sum_0 = vld1q_s32(p_sum);
                        let v_sum_1 = vld1q_s32(p_sum.add(4));
                        let v_sum_2 = vld1q_s32(p_sum.add(8));
                        let v_sum_3 = vld1q_s32(p_sum.add(12));

                        let v_sum_0 = vaddq_s32(v_sum_0, vreinterpretq_s32_u32(v_b_u32_0));
                        let v_sum_1 = vaddq_s32(v_sum_1, vreinterpretq_s32_u32(v_b_u32_1));
                        let v_sum_2 = vaddq_s32(v_sum_2, vreinterpretq_s32_u32(v_b_u32_2));
                        let v_sum_3 = vaddq_s32(v_sum_3, vreinterpretq_s32_u32(v_b_u32_3));

                        vst1q_s32(p_sum, v_sum_0);
                        vst1q_s32(p_sum.add(4), v_sum_1);
                        vst1q_s32(p_sum.add(8), v_sum_2);
                        vst1q_s32(p_sum.add(12), v_sum_3);

                        c += 16;
                    }
                }

                for c_rem in c..n {
                    col_sums_b[c_rem] += b_slice[row_off + c_rem] as i32;
                }
            }
        }

        for i in 0..m {
            for j in 0..n {
                let mut val = constant_term;
                if zp_b != 0 {
                    val -= (row_sums_a[i] as f32) * (zp_b as f32);
                }
                if zp_a != 0 {
                    val -= (col_sums_b[j] as f32) * (zp_a as f32);
                }
                out_slice[i * n + j] = val;
            }
        }

        // Process rows of B column-wise - PACKING B
        // Optimized implementation: Pack B into blocks for dot-product
        let mut j = 0;
        while j + 16 <= n {
            // Pack B for this strip of 16 columns.
            let k_aligned = k.div_ceil(4) * 4;
            let mut packed_b = vec![0u8; k_aligned * 16];

            unsafe {
                let mut k_idx = 0;
                let mut dst_ptr = packed_b.as_mut_ptr();

                while k_idx < k {
                    let rows_left = k - k_idx;

                    // Load 4 lines of B (16 cols wide)
                    let v0 = if rows_left > 0 {
                        vld1q_u8(b_slice.as_ptr().add(k_idx * n + j))
                    } else {
                        vdupq_n_u8(0)
                    };
                    let v1 = if rows_left > 1 {
                        vld1q_u8(b_slice.as_ptr().add((k_idx + 1) * n + j))
                    } else {
                        vdupq_n_u8(0)
                    };
                    let v2 = if rows_left > 2 {
                        vld1q_u8(b_slice.as_ptr().add((k_idx + 2) * n + j))
                    } else {
                        vdupq_n_u8(0)
                    };
                    let v3 = if rows_left > 3 {
                        vld1q_u8(b_slice.as_ptr().add((k_idx + 3) * n + j))
                    } else {
                        vdupq_n_u8(0)
                    };

                    // Transpose 4x16 -> 16x4 for dot product
                    let t0 = vzip1q_u8(v0, v1);
                    let t1 = vzip2q_u8(v0, v1);
                    let t2 = vzip1q_u8(v2, v3);
                    let t3 = vzip2q_u8(v2, v3);

                    let res0 = vzip1q_u16(vreinterpretq_u16_u8(t0), vreinterpretq_u16_u8(t2));
                    let res1 = vzip2q_u16(vreinterpretq_u16_u8(t0), vreinterpretq_u16_u8(t2));
                    let res2 = vzip1q_u16(vreinterpretq_u16_u8(t1), vreinterpretq_u16_u8(t3));
                    let res3 = vzip2q_u16(vreinterpretq_u16_u8(t1), vreinterpretq_u16_u8(t3));

                    vst1q_u8(dst_ptr, vreinterpretq_u8_u16(res0));
                    vst1q_u8(dst_ptr.add(16), vreinterpretq_u8_u16(res1));
                    vst1q_u8(dst_ptr.add(32), vreinterpretq_u8_u16(res2));
                    vst1q_u8(dst_ptr.add(48), vreinterpretq_u8_u16(res3));

                    dst_ptr = dst_ptr.add(64);
                    k_idx += 4;
                }
            }

            // Process Rows of A against Packed B
            for i in 0..m {
                let row_a = &a_slice[i * k..(i + 1) * k];
                let row_out = &mut out_slice[i * n..(i + 1) * n];

                unsafe {
                    let mut acc0 = vdupq_n_u32(0);
                    let mut acc1 = vdupq_n_u32(0);
                    let mut acc2 = vdupq_n_u32(0);
                    let mut acc3 = vdupq_n_u32(0);

                    let mut k_idx = 0;
                    let mut pb_ptr = packed_b.as_ptr();

                    while k_idx < k {
                        let val_ptr = row_a.as_ptr().add(k_idx) as *const u32;
                        let a_val = if k_idx + 4 <= k {
                            core::ptr::read_unaligned(val_ptr)
                        } else {
                            let mut tmp = [0u8; 4];
                            for x in 0..k.saturating_sub(k_idx).min(4) {
                                tmp[x] = row_a[k_idx + x];
                            }
                            u32::from_ne_bytes(tmp)
                        };

                        let va = vdupq_n_u32(a_val);
                        let va_u8 = vreinterpretq_u8_u32(va);

                        let vb0 = vld1q_u8(pb_ptr);
                        let vb1 = vld1q_u8(pb_ptr.add(16));
                        let vb2 = vld1q_u8(pb_ptr.add(32));
                        let vb3 = vld1q_u8(pb_ptr.add(48));

                        acc0 = vdotq_u32_custom(acc0, va_u8, vb0);
                        acc1 = vdotq_u32_custom(acc1, va_u8, vb1);
                        acc2 = vdotq_u32_custom(acc2, va_u8, vb2);
                        acc3 = vdotq_u32_custom(acc3, va_u8, vb3);

                        pb_ptr = pb_ptr.add(64);
                        k_idx += 4;
                    }

                    let ptr_out = row_out.as_mut_ptr().add(j);
                    let existing0 = vld1q_f32(ptr_out);
                    let existing1 = vld1q_f32(ptr_out.add(4));
                    let existing2 = vld1q_f32(ptr_out.add(8));
                    let existing3 = vld1q_f32(ptr_out.add(12));

                    vst1q_f32(ptr_out, vaddq_f32(existing0, vcvtq_f32_u32(acc0)));
                    vst1q_f32(ptr_out.add(4), vaddq_f32(existing1, vcvtq_f32_u32(acc1)));
                    vst1q_f32(ptr_out.add(8), vaddq_f32(existing2, vcvtq_f32_u32(acc2)));
                    vst1q_f32(ptr_out.add(12), vaddq_f32(existing3, vcvtq_f32_u32(acc3)));
                }
            }
            j += 16;
        }

        // Remainder handling
        while j < n {
            for i in 0..m {
                let row_a = &a_slice[i * k..(i + 1) * k];
                let row_out = &mut out_slice[i * n..(i + 1) * n];

                let mut sum: i32 = 0;
                for k_idx in 0..k {
                    sum += (row_a[k_idx] as i32) * (b_slice[k_idx * n + j] as i32);
                }
                row_out[j] += sum as f32;
            }
            j += 1;
        }
    }

    let mut output_shape = if batch_a >= batch_b {
        a.shape[..a_dims - 2].to_vec()
    } else {
        b.shape[..b_dims - 2].to_vec()
    };
    output_shape.push(m);
    output_shape.push(n);

    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(output_shape),
    }
}
