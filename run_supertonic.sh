#!/bin/sh
# Build and run Supertonic TTS example
# The build.rs will automatically download models and generate code from model.toml

# We use cargo run which will trigger build.rs and download missing models
cargo run --release -p supertonic-example --bin supertonic -- "The project now compiles successfully."
