import os
import json
from funasr import AutoModel


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.
    
    CER = (S + D + I) / N
    where:
        S = number of substitutions
        D = number of deletions
        I = number of insertions
        N = total number of characters in reference
    
    Args:
        reference: Ground truth text
        hypothesis: Generated text
        
    Returns:
        CER as a float between 0 and 1 (capped at 1.0/100%)
    """
    # Remove spaces for character-level comparison
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")
    
    # Dynamic programming for edit distance
    len_ref = len(ref)
    len_hyp = len(hyp)
    
    # Initialize DP table
    dp = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]
    
    # Base cases
    for i in range(len_ref + 1):
        dp[i][0] = i
    for j in range(len_hyp + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                substitution = dp[i-1][j-1] + 1
                insertion = dp[i][j-1] + 1
                deletion = dp[i-1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)
    
    # Calculate CER
    edit_distance = dp[len_ref][len_hyp]
    cer = edit_distance / len_ref if len_ref > 0 else 0.0
    
    # Cap CER at 100% (1.0)
    cer = min(cer, 1.0)
    
    return cer


reference_file = "/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/content.json"
with open(reference_file, 'r', encoding='utf-8') as f:
    references = json.load(f)

audio_pth_to_transcript = {}
for sample in references:
    audio_path = sample['audio_path']
    transcript = sample['text_yue']
    audio_pth_to_transcript[audio_path] = transcript

model_dir = "/home/duy/PlayWithMino/SimulStreaming/save_dir/WSYue-ASR/sensevoice_small_yue"

model = AutoModel(
        model=model_dir,
        device="cuda:0",
    )

# Initialize evaluation metrics
total_cer = 0.0
num_samples = 0
cer_list = []

for wav_path in os.listdir("/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/wav_"):
    wav_path = os.path.join("/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/wav_", wav_path)
    res = model.generate(
        wav_path,
        cache={},
        language="yue",
        use_itn=True,
        batch_size=64,
    )
    transcription = res[0]["text"][37:]
    reference = audio_pth_to_transcript[wav_path]
    
    # Calculate CER
    cer = calculate_cer(reference, transcription)
    total_cer += cer
    num_samples += 1
    cer_list.append(cer)
    
    print(f"Reference: {reference}")
    print(f"Transcription: {transcription}")
    print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
    print("-----")

# Print evaluation summary
print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)
print(f"Total samples: {num_samples}")
print(f"Average CER: {total_cer/num_samples:.4f} ({(total_cer/num_samples)*100:.2f}%)")
print(f"Min CER: {min(cer_list):.4f} ({min(cer_list)*100:.2f}%)")
print(f"Max CER: {max(cer_list):.4f} ({max(cer_list)*100:.2f}%)")
print("=" * 80)