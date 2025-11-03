#!/bin/bash
# Example workflow: Batch transcription with integrated evaluation
# This script demonstrates the complete workflow with automatic evaluation

set -e  # Exit on error

# Configuration
AUDIO_DIR="/data/mino/AISHELL-1/data_aishell/wav/train/S0002"
REFERENCE_FILE="/data/mino/AISHELL-1/data_aishell/reference.json"
MODEL_PATH="large-v3.pt"
OUTPUT_DIR="save_dir/streaming_largev3-zh_aishell_results/"
NUM_FILES=100

echo ""
echo "=========================================="
echo "SimulStreaming Integrated Workflow Example"
echo "=========================================="
echo ""

# Run batch transcription with automatic evaluation
echo ""
echo "Running batch transcription with automatic evaluation..."
echo "  Audio directory: $AUDIO_DIR"
echo "  Number of files: $NUM_FILES"
echo "  Reference file: $REFERENCE_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

python simulstreaming_whisper.py "$AUDIO_DIR" \
    --model_path "$MODEL_PATH" \
    --logdir "$OUTPUT_DIR" \
    --vac \
    --vac-chunk-size 2 \
    --reference-file "$REFERENCE_FILE" \
    --log-level INFO \
    --lan "zh" \
    --beams 4 \
    --frame_threshold 20 \
    --num-audios "$NUM_FILES" \

echo ""
echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - batch_transcriptions.json (all transcriptions)"
echo "  - evaluation_results.json (CER metrics)"
echo "  - *_transcription.txt (individual files)"
echo ""
echo "Quick results:"
echo "  Average CER: $(cat $OUTPUT_DIR/evaluation_results.json | grep -o '"average_cer": [0-9.]*' | cut -d' ' -f2)"
echo ""
echo "View detailed results:"
echo "  cat $OUTPUT_DIR/evaluation_results.json | jq '.per_file_results[] | {file: .file, cer: .cer}'"
echo ""
