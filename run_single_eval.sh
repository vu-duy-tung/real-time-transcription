#!/bin/bash
# Example workflow: Single file transcription with integrated evaluation
# This script demonstrates the complete workflow with automatic evaluation

set -e  # Exit on error

# Configuration
AUDIO_PATH="/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/wav_/gd0000333_43720_50830.wav"
REFERENCE_FILE="/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/content.json"
MODEL_PATH="large-v3.pt"
OUTPUT_DIR="./example_output"

echo "=========================================="
echo "SimulStreaming Integrated Workflow Example"
echo "=========================================="
echo ""

# Run single file transcription with automatic evaluation
echo "Running single file transcription with automatic evaluation..."
echo "  Audio path: $AUDIO_PATH"
echo "  Reference file: $REFERENCE_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

python simulstreaming_whisper.py "$AUDIO_PATH" \
    --model_path "$MODEL_PATH" \
    --logdir "$OUTPUT_DIR" \
    --vac \
    --vac-chunk-size 0.5 \
    --lan "yue" \
    --beams 10 \
    --reference-file "$REFERENCE_FILE" \
    --log-level DEBUG

echo ""
echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo ""

