use crate::kernels::utils;
use crate::tensor::TensorView;
use matrixmultiply::sgemm;
pub fn conv1d<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let in_shape = &input.shape;
    let w_shape = &weights.shape;
    let rank = in_shape.len();
    let (batch_size, in_channels, input_len) = if rank == 3 {
        (in_shape[0], in_shape[1], in_shape[2])
    } else if rank == 2 {
        (in_shape[0], 1, in_shape[1])
    } else {
        panic!("Conv1d: Unsupported input rank {}", rank);
    };
    let out_channels = w_shape[0];
    let kernel_size = w_shape[2];
    let dilation = if dilations.is_empty() {
        1
    } else {
        dilations[0] as usize
    };
    let stride = if strides.is_empty() {
        1
    } else {
        strides[0] as usize
    };
    let pad_left = if pads.is_empty() { 0 } else { pads[0] as usize };
    let pad_right = if pads.len() > 1 { pads[1] as usize } else { 0 };
    let output_len =
        (input_len + pad_left + pad_right - dilation * (kernel_size - 1) - 1) / stride + 1;
    let total_output_size = batch_size * out_channels * output_len;
    utils::ensure_capacity(out, total_output_size);
    unsafe {
        out.set_len(total_output_size);
    }
    let in_channels_per_group = in_channels / group as usize;
    let out_channels_per_group = out_channels / group as usize;
    let unfolded_rows = in_channels_per_group * kernel_size;
    let unfolded_size = unfolded_rows * output_len;
    let mut unfolded = vec![0.0; unfolded_size];

    // Fast path: stride=1, dilation=1, no padding - direct memory layout
    let is_fast_path =
        stride == 1 && dilation == 1 && pad_left == 0 && pad_right == 0 && kernel_size == 1;

    for b in 0..batch_size {
        for g in 0..group as usize {
            let in_group_offset = (b * in_channels + g * in_channels_per_group) * input_len;

            if is_fast_path {
                // Fast path: kernel_size=1, no padding, stride=1
                // Direct copy without im2col overhead
                let src_offset = in_group_offset;
                let copy_len = in_channels_per_group * input_len;
                unfolded[..copy_len]
                    .copy_from_slice(&input.data[src_offset..src_offset + copy_len]);
            } else {
                // Standard im2col path with optimizations
                // Only zero out what we need
                if pad_left > 0 || pad_right > 0 || dilation > 1 {
                    unfolded.fill(0.0);
                }

                for ic in 0..in_channels_per_group {
                    let in_row_offset = in_group_offset + ic * input_len;
                    let in_data = &input.data[in_row_offset..in_row_offset + input_len];

                    for k in 0..kernel_size {
                        let k_offset = k * dilation;
                        let unfolded_row_idx = ic * kernel_size + k;
                        let unfolded_row_offset = unfolded_row_idx * output_len;

                        // Optimize: calculate valid range to avoid per-element bounds checking
                        let first_valid_out = if pad_left > k_offset {
                            ((pad_left - k_offset + stride - 1) / stride).max(0)
                        } else {
                            0
                        };
                        let last_valid_out = ((input_len + pad_left - k_offset + stride - 1)
                            / stride)
                            .min(output_len);

                        if first_valid_out < last_valid_out {
                            let unf_ptr = unfolded.as_mut_ptr();
                            let in_ptr = in_data.as_ptr();

                            if stride == 1 && pad_left == 0 {
                                // Contiguous copy for stride=1, no padding
                                let src_start = k_offset;
                                let dst_start = unfolded_row_offset;
                                let copy_len = (input_len - k_offset).min(output_len);
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        in_ptr.add(src_start),
                                        unf_ptr.add(dst_start),
                                        copy_len,
                                    );
                                }
                            } else {
                                // Strided copy
                                for t_out in first_valid_out..last_valid_out {
                                    let t_in = (t_out * stride) as isize - pad_left as isize
                                        + k_offset as isize;
                                    if t_in >= 0 && (t_in as usize) < input_len {
                                        unsafe {
                                            *unf_ptr.add(unfolded_row_offset + t_out) =
                                                *in_ptr.add(t_in as usize);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let weight_group_offset =
                (g * out_channels_per_group) * in_channels_per_group * kernel_size;
            let out_group_offset = (b * out_channels + g * out_channels_per_group) * output_len;
            unsafe {
                sgemm(
                    out_channels_per_group,
                    unfolded_rows,
                    output_len,
                    1.0,
                    weights.data.as_ptr().add(weight_group_offset),
                    unfolded_rows as isize,
                    1,
                    unfolded.as_ptr(),
                    output_len as isize,
                    1,
                    0.0,
                    out.as_mut_ptr().add(out_group_offset),
                    output_len as isize,
                    1,
                );
            }
        }
    }

    // Optimized bias addition with SIMD-friendly loop
    if let Some(b_vec) = bias {
        let out_ptr = out.as_mut_ptr();
        for b in 0..batch_size {
            for oc in 0..out_channels {
                let start = (b * out_channels + oc) * output_len;
                let b_val = b_vec.data[oc];

                // Vectorizable loop - compiler can auto-vectorize this
                unsafe {
                    for i in 0..output_len {
                        *out_ptr.add(start + i) += b_val;
                    }
                }
            }
        }
    }

    TensorView::from_slice(out, vec![batch_size, out_channels, output_len])
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorView;
    #[test]
    fn test_conv1d_grouped() {
        let input_data = vec![1.0; 6];
        let input = TensorView::from_slice(&input_data, vec![1, 2, 3]);
        let weight_data = vec![1.0; 2];
        let weights = TensorView::from_slice(&weight_data, vec![2, 1, 1]);
        let mut out = Vec::new();
        let res = conv1d(&input, &weights, None, &[1], 2, &[0, 0], &[1], &mut out);
        assert_eq!(res.shape, vec![1, 2, 3]);
        assert_eq!(res.data, vec![1.0; 6]);
    }
    #[test]
    fn test_conv1d_simple() {
        let input_data = vec![1.0, 2.0, 3.0];
        let input = TensorView::from_slice(&input_data, vec![1, 1, 3]);
        let weight_data = vec![1.0, 1.0];
        let weights = TensorView::from_slice(&weight_data, vec![1, 1, 2]);
        let mut out = Vec::new();
        let res = conv1d(&input, &weights, None, &[1], 1, &[0, 0], &[1], &mut out);
        assert_eq!(res.shape, vec![1, 1, 2]);
        assert_eq!(res.data, vec![3.0, 5.0]);
    }
}
