use crate::kernels::activations::{sigmoid, tanh};
use crate::tensor::TensorView;

pub fn lstm<'b, 'a>(
    input: &TensorView<'b>,
    w: &TensorView<'b>,
    r: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    _sequence_lens: Option<&TensorView<'b>>,
    initial_h: Option<&TensorView<'b>>,
    initial_c: Option<&TensorView<'b>>,
    out_y: &'a mut Vec<f32>,
    out_h: &'a mut Vec<f32>,
    out_c: &'a mut Vec<f32>,
) -> (TensorView<'a>, TensorView<'a>, TensorView<'a>) {
    let seq_len = input.shape[0];
    let batch_size = input.shape[1];
    let input_size = input.shape[2];

    let num_directions = w.shape[0];
    let hidden_size = w.shape[1] / 4;
    if num_directions != 1 {
        panic!("LSTM: Only num_directions=1 supported");
    }
    if batch_size != 1 {
        panic!("LSTM: Only batch_size=1 supported");
    }
    let w_data = &w.data;
    let r_data = &r.data;
    let default_bias = vec![0.0; 8 * hidden_size];
    let bias_data: &[f32] = if let Some(b) = bias {
        b.data.as_ref()
    } else {
        default_bias.as_slice()
    };
    let (bias_w_slice, bias_r_slice) = bias_data.split_at(4 * hidden_size);
    
    out_h.resize(hidden_size, 0.0);
    if let Some(h) = initial_h {
        out_h.copy_from_slice(&h.data);
    } else {
        out_h.fill(0.0);
    }
    out_c.resize(hidden_size, 0.0);
    if let Some(c) = initial_c {
        out_c.copy_from_slice(&c.data);
    } else {
        out_c.fill(0.0);
    }
    out_y.resize(seq_len * hidden_size, 0.0);
    
    let mut gates = vec![0.0; 4 * hidden_size];
    let mut w_contribution = vec![0.0; 4 * hidden_size];
    let mut r_contribution = vec![0.0; 4 * hidden_size];
    
    for t in 0..seq_len {
        let input_offset = t * input_size;
        let x_t = &input.data[input_offset..input_offset + input_size];
        
        // Use GEMM for W @ x_t: [4*hidden, input_size] @ [input_size, 1] = [4*hidden, 1]
        unsafe {
            matrixmultiply::sgemm(
                4 * hidden_size,  // m
                input_size,       // k
                1,                // n
                1.0,
                w_data.as_ptr(),
                input_size as isize,
                1,
                x_t.as_ptr(),
                1,
                1,
                0.0,
                w_contribution.as_mut_ptr(),
                1,
                1,
            );
        }
        
        // Use GEMM for R @ h_t: [4*hidden, hidden] @ [hidden, 1] = [4*hidden, 1]
        unsafe {
            matrixmultiply::sgemm(
                4 * hidden_size,  // m
                hidden_size,      // k
                1,                // n
                1.0,
                r_data.as_ptr(),
                hidden_size as isize,
                1,
                out_h.as_ptr(),
                1,
                1,
                0.0,
                r_contribution.as_mut_ptr(),
                1,
                1,
            );
        }
        
        // Combine: gates = W@x + R@h + bias
        for g in 0..(4 * hidden_size) {
            gates[g] = w_contribution[g] + r_contribution[g] + bias_w_slice[g] + bias_r_slice[g];
        }
        
        // Apply activations and update cell/hidden state
        for k in 0..hidden_size {
            let i_gate = sigmoid(gates[k]);
            let o_gate = sigmoid(gates[hidden_size + k]);
            let f_gate = sigmoid(gates[2 * hidden_size + k]);
            let c_gate = tanh(gates[3 * hidden_size + k]);
            let ct = f_gate * out_c[k] + i_gate * c_gate;
            let ht = o_gate * tanh(ct);
            out_c[k] = ct;
            out_h[k] = ht;
            out_y[t * hidden_size + k] = ht;
        }
    }
    let output_y_shape = vec![seq_len, 1, 1, hidden_size];
    let output_y = TensorView::from_slice(out_y, output_y_shape);
    let output_h_shape = vec![1, 1, hidden_size];
    let output_h = TensorView::from_slice(out_h, output_h_shape);
    let output_c_shape = vec![1, 1, hidden_size];
    let output_c = TensorView::from_slice(out_c, output_c_shape);
    (output_y, output_h, output_c)
}
