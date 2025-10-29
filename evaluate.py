#!/usr/bin/env python3
"""
Evaluation module for transcription quality using CER (Character Error Rate).
Can be used as a library or standalone script.
"""

import argparse
import json
import os
from pathlib import Path
import logging

# Module logger - will use the parent logger when imported
logger = logging.getLogger(__name__)


def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER) between reference and hypothesis strings.

    CER = min(1.0, (S + D + I) / N)
    where:
        S = number of substitutions
        D = number of deletions
        I = number of insertions
        N = number of characters in reference

    Returns CER clipped to at most 1.0 (100%).
    """
    # Normalize strings
    ref = reference.strip()
    hyp = hypothesis.strip()

    # Handle edge cases: define CER=0 if both empty, else 1.0 when reference empty
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    # Initialize matrix for dynamic programming
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # deletions
    for j in range(n + 1):
        dp[0][j] = j  # insertions

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]  # no operation needed
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + 1     # substitution
                )

    # Calculate CER and clamp to a maximum of 1.0
    edit_distance = dp[m][n]
    cer = edit_distance / len(ref)
    return min(cer, 1.0)


def load_references(reference_file, language='en'):
    """
    Load reference transcriptions from JSON file.
    
    Args:
        reference_file (str): Path to JSON file containing references
        language (str): Language code for reference text field (default: 'en')
    
    Returns:
        dict: Dictionary mapping audio file basenames to reference texts
    """
    if language == 'auto' or language == 'zh':
        language = 'yue'
        
    with open(reference_file, 'r', encoding='utf-8') as f:
        references = json.load(f)
    
    # Create mapping from audio filename to reference text
    ref_map = {}
    text_field = f'text_{language}'
    
    for item in references:
        audio_path = item.get('audio_path', '')
        basename = os.path.basename(audio_path)
        # Remove extension for matching
        basename_noext = os.path.splitext(basename)[0]
        
        # Get reference text for specified language
        ref_text = item.get(text_field, '')
        
        if ref_text:
            ref_map[basename] = ref_text
            ref_map[basename_noext] = ref_text
    
    logger.info(f"Loaded {len(ref_map)} reference transcriptions for language '{language}'")
    return ref_map


def load_generated_transcriptions(transcription_file):
    """
    Load generated transcriptions from JSON file (batch output).
    
    Args:
        transcription_file (str): Path to batch_transcriptions.json
    
    Returns:
        dict: Dictionary mapping filenames to generated texts
    """
    with open(transcription_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gen_map = {}
    for result in data.get('results', []):
        filename = result.get('file', '')
        text = result.get('transcription', '')
        
        basename_noext = os.path.splitext(filename)[0]
        gen_map[filename] = text
        gen_map[basename_noext] = text
    
    logger.info(f"Loaded {len(gen_map)} generated transcriptions")
    return gen_map


def load_individual_transcriptions(logdir):
    """
    Load individual transcription files from directory.
    
    Args:
        logdir (str): Directory containing *_transcription.txt files
    
    Returns:
        dict: Dictionary mapping filenames to generated texts
    """
    gen_map = {}
    logdir_path = Path(logdir)
    
    for txt_file in logdir_path.glob('*_transcription.txt'):
        # Extract original filename
        filename = txt_file.stem.replace('_transcription', '')
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        gen_map[filename] = text
        # Also store with .wav extension for matching
        gen_map[f"{filename}.wav"] = text
    
    logger.info(f"Loaded {len(gen_map)} individual transcription files")
    return gen_map


def evaluate_transcriptions(references, generated):
    """
    Evaluate generated transcriptions against references using CER.
    
    Args:
        references (dict): Reference transcriptions
        generated (dict): Generated transcriptions
    
    Returns:
        dict: Evaluation results including per-file CER and average
    """
    results = []
    total_cer = 0.0
    matched_count = 0
    
    for filename, ref_text in references.items():
        if filename in generated:
            gen_text = generated[filename]
            cer = calculate_cer(ref_text, gen_text)
            
            results.append({
                'file': filename,
                'reference': ref_text,
                'generated': gen_text,
                'cer': cer,
                'ref_length': len(ref_text),
                'gen_length': len(gen_text)
            })
            
            total_cer += cer
            matched_count += 1
            
            logger.debug(f"{filename}: CER={cer:.4f}")
    
    # Calculate average CER
    avg_cer = total_cer / matched_count if matched_count > 0 else 0.0
    
    summary = {
        'total_files': len(references),
        'matched_files': matched_count,
        'unmatched_files': len(references) - matched_count,
        'average_cer': avg_cer,
        'per_file_results': results
    }
    
    return summary


def main():
    # Setup logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
    
    parser = argparse.ArgumentParser(
        description='Evaluate transcription quality using CER metric'
    )
    
    parser.add_argument(
        'reference_file',
        type=str,
        help='Path to JSON file containing reference transcriptions (with audio_path and text_<language> fields)'
    )
    
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='Directory containing generated transcriptions (batch_transcriptions.json or individual files)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results JSON (default: <logdir>/evaluation_results.json)'
    )
    
    parser.add_argument(
        '--batch-file',
        type=str,
        default='batch_transcriptions.json',
        help='Name of batch transcription file (default: batch_transcriptions.json)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language code for reference text field (default: en)'
    )
    
    parser.add_argument(
        '-l', '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the log level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(args.log_level)
    
    # Load references
    logger.info(f"Loading references from: {args.reference_file}")
    references = load_references(args.reference_file, language=args.language)
    
    # Load generated transcriptions
    logger.info(f"Loading generated transcriptions from: {args.logdir}")
    
    # Try to load batch file first
    batch_file = os.path.join(args.logdir, args.batch_file)
    if os.path.exists(batch_file):
        logger.info(f"Found batch transcription file: {batch_file}")
        generated = load_generated_transcriptions(batch_file)
    else:
        logger.info(f"Batch file not found, loading individual transcription files")
        generated = load_individual_transcriptions(args.logdir)
    
    if not generated:
        logger.error("No generated transcriptions found!")
        return 1
    
    # Evaluate
    logger.info("Evaluating transcriptions...")
    evaluation_results = evaluate_transcriptions(references, generated)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total reference files: {evaluation_results['total_files']}")
    print(f"Matched files: {evaluation_results['matched_files']}")
    print(f"Unmatched files: {evaluation_results['unmatched_files']}")
    print(f"\nAverage CER: {evaluation_results['average_cer']:.4f} ({evaluation_results['average_cer']*100:.2f}%)")
    print("=" * 80)
    
    # Print per-file results
    if args.log_level == 'DEBUG' or args.log_level == 'INFO':
        print("\nPer-file results:")
        print("-" * 80)
        for result in evaluation_results['per_file_results']:
            print(f"{result['file']:40s} CER: {result['cer']:.4f} ({result['cer']*100:.2f}%)")
    
    # Save results
    output_file = args.output or os.path.join(args.logdir, 'evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation results saved to: {output_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())
