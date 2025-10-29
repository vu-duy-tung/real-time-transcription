#!/bin/bash
# Compare different Cantonese Whisper models on the same dataset

AUDIO_DIR="save_dir/data/WSYue-ASR-eval/Short/wav_"
REF_FILE="save_dir/data/WSYue-ASR-eval/Short/content.json"

echo "=================================================="
echo "Comparing Cantonese Whisper Models"
echo "=================================================="
echo ""

# Model 1: khleeloo/whisper-large-v3-cantonese
echo "Testing Model 1: khleeloo/whisper-large-v3-cantonese"
python whisper_offline/whisper_offline.py \
    --audio_dir "$AUDIO_DIR" \
    --reference_file "$REF_FILE" \
    --output_dir results/offline_khleeloo \
    --model_name khleeloo/whisper-large-v3-cantonese

echo ""
echo "=================================================="
echo ""

# Model 2: JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english
echo "Testing Model 2: JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english"
python whisper_offline/whisper_offline.py \
    --audio_dir "$AUDIO_DIR" \
    --reference_file "$REF_FILE" \
    --output_dir results/offline_jacky \
    --model_name JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english

echo ""
echo "=================================================="
echo "Comparison Complete!"
echo "=================================================="
echo ""
echo "Results:"
echo "  Model 1 (khleeloo):  results/offline_khleeloo/evaluation_results.json"
echo "  Model 2 (JackyHoCL): results/offline_jacky/evaluation_results.json"
echo ""
echo "To view results:"
echo "  cat results/offline_khleeloo/evaluation_results.json | jq '.average_cer'"
echo "  cat results/offline_jacky/evaluation_results.json | jq '.average_cer'"
