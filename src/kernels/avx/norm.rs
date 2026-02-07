#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn layer_norm_x86(
    input: *const f32,
    scale: *const f32,
    bias: *const f32,
    output: *mut f32,
    norm_size: usize,
    outer_size: usize,
    epsilon: f32,
) {
    unsafe {
        for i in 0..outer_size {
            let offset = i * norm_size;
            let in_ptr = input.add(offset);
            let out_ptr = output.add(offset);

            // 1. Mean
            let mut sum_v = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= norm_size {
                let v = _mm256_loadu_ps(in_ptr.add(j));
                sum_v = _mm256_add_ps(sum_v, v);
                j += 8;
            }
            // Vertical sum of sum_v
            let temp = _mm_add_ps(
                _mm256_castps256_ps128(sum_v),
                _mm256_extractf128_ps(sum_v, 1),
            );
            let temp = _mm_add_ps(temp, _mm_movehl_ps(temp, temp));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            let mut sum = _mm_cvtss_f32(temp);

            // Tail
            while j < norm_size {
                sum += *in_ptr.add(j);
                j += 1;
            }
            let mean = sum / norm_size as f32;
            let mean_v = _mm256_set1_ps(mean);

            // 2. Variance
            let mut sum_sq_v = _mm256_setzero_ps();
            j = 0;
            while j + 8 <= norm_size {
                let v = _mm256_loadu_ps(in_ptr.add(j));
                let diff = _mm256_sub_ps(v, mean_v);
                sum_sq_v = _mm256_fmadd_ps(diff, diff, sum_sq_v);
                j += 8;
            }
            let temp = _mm_add_ps(
                _mm256_castps256_ps128(sum_sq_v),
                _mm256_extractf128_ps(sum_sq_v, 1),
            );
            let temp = _mm_add_ps(temp, _mm_movehl_ps(temp, temp));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            let mut sum_sq = _mm_cvtss_f32(temp);

            while j < norm_size {
                let diff = *in_ptr.add(j) - mean;
                sum_sq += diff * diff;
                j += 1;
            }

            let var = sum_sq / norm_size as f32;
            let inv_std = 1.0 / (var + epsilon).sqrt();
            let inv_std_v = _mm256_set1_ps(inv_std);

            // 3. Normalize & Scale & Shift
            j = 0;
            while j + 8 <= norm_size {
                let v_in = _mm256_loadu_ps(in_ptr.add(j));
                let v_gamma = _mm256_loadu_ps(scale.add(j));
                let v_beta = _mm256_loadu_ps(bias.add(j));

                let v_norm = _mm256_sub_ps(v_in, mean_v);
                let v_scaled = _mm256_mul_ps(v_norm, inv_std_v);
                let v_res = _mm256_fmadd_ps(v_scaled, v_gamma, v_beta);

                _mm256_storeu_ps(out_ptr.add(j), v_res);
                j += 8;
            }
            while j < norm_size {
                let val = *in_ptr.add(j);
                *out_ptr.add(j) = (val - mean) * inv_std * *scale.add(j) + *bias.add(j);
                j += 1;
            }
        }
    }
}
