#!/bin/sh

# Build and run Silero VAD example with the new workspace structure
# The build.rs will automatically generate code from model.onnx

cargo run --release -p silero-example --bin silero -- fixtures/zh.wav
