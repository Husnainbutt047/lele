#!/bin/sh
cargo run --bin lele_gen examples/sensevoice/sensevoice.int8.onnx examples/sensevoice/src SenseVoice
cargo run -r -p sensevoice-example --bin sensevoice -- fixtures/zh.wav
