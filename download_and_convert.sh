#!/bin/bash
# Download and convert the Cantonese Whisper model

echo "=========================================="
echo "Downloading and Converting Cantonese Model"
echo "=========================================="
echo ""
echo "This will:"
echo "1. Download JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english from HuggingFace"
echo "2. Convert it to OpenAI Whisper format"
echo "3. Save as ./cantonese-yue-en.pt"
echo ""
echo "This may take a few minutes depending on your internet speed..."
echo ""

python convert_hf_to_whisper.py \
    --hf-model khleeloo/whisper-large-v3-cantonese \
    --output-path ./khleeloo-large-v3.pt

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Conversion Complete!"
    echo "=========================================="
    echo "Model saved to: ./cantonese-yue-en.pt"
    echo ""
    echo "You can now use it with:"
    echo "  ./run_cantonese_whisper.sh <audio_file>"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Conversion Failed"
    echo "=========================================="
    echo "Please check the error messages above."
    exit 1
fi
