use crate::kernels::utils;
use crate::tensor::TensorView;
use std::borrow::Cow;
pub fn softmax<'b, 'a>(
    input: &TensorView<'b>,
    axis: i32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    assert!(axis < ndim);
    let numel = input.data.len();
    utils::ensure_capacity(out_buf, numel);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), numel) };
    let inner_size: usize = input.shape[axis + 1..].iter().product();
    let axis_size = input.shape[axis];
    let outer_size: usize = input.shape[..axis].iter().product();
    let data = &input.data;
    if inner_size == 1 {
        #[cfg(target_arch = "x86_64")]
        {
            for i in 0..outer_size {
                let start = i * axis_size;
                let end = start + axis_size;
                let src = &data[start..end];
                let dst = &mut out_slice[start..end];
                unsafe {
                    softmax_avx2(src, dst);
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..outer_size {
                let start = i * axis_size;
                let end = start + axis_size;
                let src = &data[start..end];
                let dst = &mut out_slice[start..end];
                let max_val = src.iter().fold(f32::MIN, |a, &b| a.max(b));
                let mut sum = 0.0;
                for (j, &val) in src.iter().enumerate() {
                    let e = (val - max_val).exp();
                    dst[j] = e;
                    sum += e;
                }
                let inv_sum = 1.0 / sum;
                for x in dst.iter_mut() {
                    *x *= inv_sum;
                }
            }
        }
    } else {
        unimplemented!("Softmax only supported on last dimension for now");
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: std::borrow::Cow::Owned(input.shape.to_vec()),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn softmax_avx2(src: &[f32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let len = src.len();
        let simd_end = (len / 8) * 8;
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();

        // 1. Find max
        let mut max_vec = _mm256_set1_ps(f32::MIN);
        let mut j = 0;
        while j < simd_end {
            let v = _mm256_loadu_ps(src_ptr.add(j));
            max_vec = _mm256_max_ps(max_vec, v);
            j += 8;
        }
        let mut max_val = {
            let hi = _mm256_extractf128_ps(max_vec, 1);
            let lo = _mm256_castps256_ps128(max_vec);
            let m128 = _mm_max_ps(lo, hi);
            let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
            _mm_cvtss_f32(m32)
        };
        for k in simd_end..len {
            max_val = max_val.max(*src_ptr.add(k));
        }

        // 2. exp(x - max) and sum
        let max_broadcast = _mm256_set1_ps(max_val);
        let mut sum_vec = _mm256_setzero_ps();
        j = 0;
        while j < simd_end {
            let v = _mm256_loadu_ps(src_ptr.add(j));
            let shifted = _mm256_sub_ps(v, max_broadcast);
            let exp_val = crate::kernels::avx::math::avx2_exp_ps(shifted);
            _mm256_storeu_ps(dst_ptr.add(j), exp_val);
            sum_vec = _mm256_add_ps(sum_vec, exp_val);
            j += 8;
        }
        let mut sum = crate::kernels::avx::math::hsum_ps(sum_vec);
        for k in simd_end..len {
            let e = (*src_ptr.add(k) - max_val).exp();
            *dst_ptr.add(k) = e;
            sum += e;
        }

        // 3. Normalize
        let inv_sum = 1.0 / sum;
        let inv_sum_vec = _mm256_set1_ps(inv_sum);
        j = 0;
        while j < simd_end {
            let v = _mm256_loadu_ps(dst_ptr.add(j));
            let normalized = _mm256_mul_ps(v, inv_sum_vec);
            _mm256_storeu_ps(dst_ptr.add(j), normalized);
            j += 8;
        }
        for k in simd_end..len {
            *dst_ptr.add(k) *= inv_sum;
        }
    }
}
pub fn layer_norm<'b, 'a>(
    input: &TensorView<'b>,
    scale: &TensorView<'b>,
    bias: &TensorView<'b>,
    axis: i32,
    epsilon: f32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    let outer_size: usize = input.shape[..axis].iter().product();
    let norm_size: usize = input.shape[axis..].iter().product();
    utils::ensure_capacity(out_buf, input.data.len());
    let out_slice =
        unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), input.data.len()) };
    #[cfg(target_arch = "x86_64")]
    unsafe {
        crate::kernels::avx::norm::layer_norm_x86(
            input.data.as_ptr(),
            scale.data.as_ptr(),
            bias.data.as_ptr(),
            out_buf.as_mut_ptr(),
            norm_size,
            outer_size,
            epsilon,
        );
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let src = &input.data;
        let gamma = &scale.data;
        let beta = &bias.data;

        for i in 0..outer_size {
            let offset = i * norm_size;
            let chunk = &src[offset..offset + norm_size];
            let out_chunk = &mut out_slice[offset..offset + norm_size];
            let sum: f32 = chunk.iter().sum();
            let mean = sum / norm_size as f32;
            let var_sum: f32 = chunk.iter().map(|&x| (x - mean) * (x - mean)).sum();
            let var = var_sum / norm_size as f32;
            let inv_std = 1.0 / (var + epsilon).sqrt();
            for j in 0..norm_size {
                out_chunk[j] = (chunk[j] - mean) * inv_std * gamma[j] + beta[j];
            }
        }
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: std::borrow::Cow::Owned(input.shape.to_vec()),
    }
}
pub fn batch_norm<'b, 'a>(
    input: &TensorView<'b>,
    scale: &TensorView<'b>,
    bias: &TensorView<'b>,
    mean: &TensorView<'b>,
    var: &TensorView<'b>,
    epsilon: f32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let shape = &input.shape;
    let numel = input.data.len();
    utils::ensure_capacity(out_buf, numel);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), numel) };
    if shape.len() == 4 {
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let spatial_size = h * w;
        let src = &input.data;
        let s = &scale.data;
        let b = &bias.data;
        let m = &mean.data;
        let v = &var.data;
        for ni in 0..n {
            for ci in 0..c {
                let offset = (ni * c + ci) * spatial_size;
                let scale_val = s[ci] / (v[ci] + epsilon).sqrt();
                let bias_val = b[ci] - m[ci] * scale_val;
                for i in 0..spatial_size {
                    out_slice[offset + i] = src[offset + i] * scale_val + bias_val;
                }
            }
        }
    } else {
        let c = if shape.len() > 1 { shape[1] } else { shape[0] };
        let outer_size = shape[0];
        let inner_size: usize = if shape.len() > 2 {
            shape[2..].iter().product()
        } else {
            1
        };
        let src = &input.data;
        let s = &scale.data;
        let b = &bias.data;
        let m = &mean.data;
        let v = &var.data;
        for i in 0..outer_size {
            for j in 0..c {
                let scale_val = s[j] / (v[j] + epsilon).sqrt();
                let bias_val = b[j] - m[j] * scale_val;
                for k in 0..inner_size {
                    let idx = (i * c + j) * inner_size + k;
                    out_slice[idx] = src[idx] * scale_val + bias_val;
                }
            }
        }
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: Cow::Owned(shape.to_vec()),
    }
}
