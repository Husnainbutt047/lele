// examples/silero/benchmark.rs

mod silerovad;

use lele::tensor::TensorView;
use std::time::Instant;

fn main() {
    println!("Benchmarking Silero VAD...");

    // 1. Load Weights
    let bin = std::fs::read("examples/silero/silerovad_weights.bin")
        .or_else(|_| std::fs::read("silerovad_weights.bin"))
        .expect("Failed to load weights");
    let model = silerovad::SileroVad::new(&bin);

    // 2. Prepare Inputs
    let input_len = 512;
    // Real audio would be better, but zeros works for perf testing
    let input_data = vec![0.0f32; input_len];
    let input_shape = [1, input_len];

    // State: [2, 1, 128]
    let state_len = 2 * 1 * 128;
    let mut state_data = vec![0.0f32; state_len];
    let state_shape = [2, 1, 128];

    // SR: [1]
    let sr_data = vec![16000i64];
    let sr_shape = [1];

    // Warmup
    println!("Warmup...");
    for _ in 0..100 {
        let input = TensorView::new(&input_data, &input_shape);
        let state = TensorView::new(&state_data, &state_shape);
        let sr = TensorView::new(&sr_data, &sr_shape);

        let (_, new_state_view) = model.forward(input, state, sr);
        // Copy back state (simulation of streaming)
        // In a real app we might swap buffers, but here copy is fine for micro-benchmark of the model
        // Break lifetime tie by consuming the view's data
        let new_state_vec = new_state_view.data.into_owned();
        state_data.copy_from_slice(&new_state_vec);
    }

    // Benchmark
    let iterations = 1000;
    println!("Running {} iterations...", iterations);

    let start = Instant::now();
    for _ in 0..iterations {
        let input = TensorView::new(&input_data, &input_shape);
        let state = TensorView::new(&state_data, &state_shape);
        let sr = TensorView::new(&sr_data, &sr_shape);

        let (_, new_state_view) = model.forward(input, state, sr);
        // We accumulate side effect to prevent optimization?
        // Rust usually doesn't optimize away FFI/complex calls unless pure.
        // But updating state simulates real usage.
        let new_state_vec = new_state_view.data.into_owned();
        state_data.copy_from_slice(&new_state_vec);
    }
    let duration = start.elapsed();

    let avg = duration.as_secs_f64() * 1000.0 / iterations as f64;
    let rtf = avg / 32.0; // 512 samples @ 16k = 32ms. RTF = processing_time / audio_time.

    println!("Total time: {:?}", duration);
    println!("Avg latency: {:.3} ms / 32ms chunk", avg);
    println!("Real Time Factor (RTF): {:.4} (lower is better)", rtf);
}
