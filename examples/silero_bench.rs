// Standalone benchmark for Silero VAD performance analysis

use lele::tensor::TensorView;
use std::time::Instant;

// Include the generated model code
#[path = "silero/silerovad.rs"]
mod silerovad;

fn main() {
    println!("=== Silero VAD Performance Analysis ===\n");

    // 1. Load Weights
    let bin = std::fs::read("examples/silero/silerovad_weights.bin")
        .or_else(|_| std::fs::read("silerovad_weights.bin"))
        .expect("Failed to load weights");
    let model = silerovad::SileroVad::new(&bin);

    // 2. Prepare Inputs (same as ORT benchmark)
    let input_len = 512;
    let input_data = vec![0.0f32; input_len];
    let input_shape = [1, input_len];

    // State: [2, 1, 128]
    let state_len = 2 * 1 * 128;
    let mut state_data = vec![0.0f32; state_len];
    let state_shape = [2, 1, 128];

    // SR: [1]
    let sr_data = vec![16000.0f32];
    let sr_shape = [1];

    // Warmup
    println!("Warmup (100 iterations)...");
    for _ in 0..100 {
        let input = TensorView::new(&input_data, &input_shape);
        let state = TensorView::new(&state_data, &state_shape);
        let sr = TensorView::new(&sr_data, &sr_shape);

        let (_, new_state_view) = model.forward(input, state, sr);
        let new_state_vec = new_state_view.data.into_owned();
        state_data.copy_from_slice(&new_state_vec);
    }

    // Benchmark
    let iterations = 1000;
    println!("Running {} iterations...\n", iterations);

    let start = Instant::now();
    for _ in 0..iterations {
        let input = TensorView::new(&input_data, &input_shape);
        let state = TensorView::new(&state_data, &state_shape);
        let sr = TensorView::new(&sr_data, &sr_shape);

        let (_, new_state_view) = model.forward(input, state, sr);
        let new_state_vec = new_state_view.data.into_owned();
        state_data.copy_from_slice(&new_state_vec);
    }
    let duration = start.elapsed();

    let avg_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;
    let rtf = avg_ms / 32.0; // 512 samples @ 16k = 32ms

    println!("=== Results ===");
    println!("Total time: {:?}", duration);
    println!("Avg latency: {:.3} ms / 32ms chunk", avg_ms);
    println!("RTF (lele): {:.4}", rtf);
    println!("\n=== Comparison with ORT ===");
    println!("ORT RTF: 0.0025");
    println!("lele RTF: {:.4}", rtf);
    println!("Performance gap: {:.1}x slower", rtf / 0.0025);
    println!("\nTarget: Achieve RTF < 0.0025 (match or beat ORT)");
}
