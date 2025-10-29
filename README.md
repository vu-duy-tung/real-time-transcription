# Real-Time Transcription

## Installation

`bash
conda create -n streamingasr python=3.10
conda activate streamingasr
bash install.sh
`

## Data installation
Download WSYue-ASR-eval for testing
`
git clone https://huggingface.co/datasets/ASLP-lab/WSYue-ASR-eval 
tar -xzf WSYue-ASR-eval/Short/wav.tar.gz
`
Preprocess the data
`
python wsyue_asr_eval.py \
        --input ./WSYue-ASR-eval/Short/content.txt \
        --output ./WSYue-ASR-eval/Short/content.json \
        --audio-dir ./WSYue-ASR-eval/Short/wav_ \
`


## Inference

- Inference of whisper-large-v3 model with SimulStreaming on a single `.wav` file
`bash
bash run_single_eval.sh
`

- Inference of whisper-large-v3 model with SimulStreaming on a folder of `.wav` files
`bash
bash run_batch_eval_wsyue.sh
`