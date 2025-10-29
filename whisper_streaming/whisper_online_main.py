#!/usr/bin/env python3

# This code is retrieved from the original WhisperStreaming whisper_online.py .
# It is refactored and simplified. Only the code that is needed for the 
# SimulWhisper backend is kept. 


import sys
import json 
import numpy as np
import librosa
from functools import lru_cache
import time
import logging


logger = logging.getLogger(__name__)

@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]

def processor_args(parser):
    """shared args for the online processors
    parser: argparse.ArgumentParser object
    """
    group = parser.add_argument_group("WhisperStreaming processor arguments (shared for simulation from file and for the server)")
    group.add_argument('--min-chunk-size', type=float, default=1.2, 
                        help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter '
                        'time, it waits, otherwise it processes the whole segment that was received by this time.')

    group.add_argument('--lan', '--language', type=str, default="en", 
                        help="Source language code, e.g. en, de, cs, or auto for automatic language detection from speech.")
    group.add_argument('--task', type=str, default='transcribe', 
                        choices=["transcribe","translate"],
                        help="Transcribe or translate.")

    group.add_argument('--vac', action="store_true", default=False, 
                        help='Use VAC = voice activity controller. Recommended. Requires torch.')
    group.add_argument('--vac-chunk-size', type=float, default=0.04, 
                        help='VAC sample size in seconds.')

    parser.add_argument("-l", "--log-level", dest="log_level", 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help="Set the log level", default='DEBUG')

    parser.add_argument("--logdir", help="Directory to save audio segments and generated texts for debugging.",
                       default=None)

def asr_factory(args, factory=None):
    """
    Creates and configures an asr and online processor object through factory that is implemented in the backend.
    """
#    if backend is None:
#        backend = args.backend
#    if backend == "simul-whisper":
#        from simul_whisper_backend import simul_asr_factory
    asr, online = factory(args)

    # Create the OnlineASRProcessor
    if args.vac:
        from whisper_streaming.vac_online_processor import VACOnlineASRProcessor
        online = VACOnlineASRProcessor(args.min_chunk_size, online)

    if args.task == "translate":
        if args.model_path.endswith(".en.pt"):
            logger.error(f"The model {args.model_path} is English only. Translation is not available. Terminating.")
            sys.exit(1)
        asr.set_translate_task()

    return asr, online

def set_logging(args,logger):
    logging.basicConfig(
        # this format would include module name:
        #    format='%(levelname)s\t%(name)s\t%(message)s')
            format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("simul_whisper").setLevel(args.log_level)
    logging.getLogger("whisper_streaming").setLevel(args.log_level)


def simulation_args(parser):
    simulation_group = parser.add_argument_group("Arguments for simulation from file")
    simulation_group.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, or directory containing multiple wav files for batch inference.")
    simulation_group.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    # TODO: offline mode is not implemented in SimulStreaming yet
#    simulation_group.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    simulation_group.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')
    simulation_group.add_argument('--batch', action="store_true", default=False, help='Enable batch processing for directory input.')
    simulation_group.add_argument('--audio-extensions', type=str, default='wav,mp3,flac,m4a', help='Comma-separated list of audio file extensions to process in batch mode.')
    simulation_group.add_argument('--num-audios', type=int, default=None, help='Limit the number of audio files to process (useful for testing). Process all files if not specified.')
    
    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation arguments")
    eval_group.add_argument('--reference-file', type=str, default=None, help='Path to JSON file containing reference transcriptions (with audio_path and text_en fields). If provided, evaluation will be performed automatically.')
    eval_group.add_argument('--eval-output', type=str, default=None, help='Path to save evaluation results JSON (default: <logdir>/evaluation_results.json)')


def get_audio_files(path, extensions):
    """Get list of audio files from a path (file or directory)."""
    import os
    
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        audio_files = []
        ext_list = [ext.strip().lower() for ext in extensions.split(',')]
        for root, dirs, files in os.walk(path):
            for file in sorted(files):
                if any(file.lower().endswith(f'.{ext}') for ext in ext_list):
                    audio_files.append(os.path.join(root, file))
        return audio_files
    else:
        raise ValueError(f"Path does not exist: {path}")


def process_single_audio_file(audio_path, args, asr, online, min_chunk, factory):
    """Process a single audio file and return transcriptions."""
    import os
    
    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path))/SAMPLING_RATE
    logger.info(f"Processing: {os.path.basename(audio_path)} - Duration: {duration:.2f}s")

    # # Reset the online processor for new file
    # asr, online = asr_factory(args, factory)
    # if args.vac:
    #     min_chunk = args.vac_chunk_size
    # else:
    #     min_chunk = args.min_chunk_size
    
    # # Warm up the ASR with first file
    # a = load_audio_chunk(audio_path, 0, 1)
    # asr.warmup(a)
    
    beg = args.start_at
    start_time = time.time()  # Actual start time for tracking purposes
    start = start_time - beg  # Offset start for elapsed time calculation
    
    # List to store all transcription segments for this file
    all_transcriptions = []

    def output_transcript(iteration_output, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time() - start

        if iteration_output:
            start_ts = iteration_output['start']
            end_ts = iteration_output['end']
            text = iteration_output['text']
            logger.debug(f"{now * 1000:.4f} {start_ts * 1000:.0f} {end_ts * 1000:.0f} {text}")
            print(f"{now * 1000:.4f} {start_ts * 1000:.0f} {end_ts * 1000:.0f} {text}", flush=True)
            
            # Store the transcription segment
            all_transcriptions.append({
                'start': start_ts,
                'end': end_ts,
                'text': text.strip()
            })
        else:
            logger.debug("No text in this segment")

    if args.offline: ## offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter(start_time=start_time)
        except AssertionError as e:
            logger.error(f"assertion error: {repr(e)}")
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode 
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path, beg, end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter(start_time=start_time)
            except AssertionError as e:
                logger.error(f"assertion error: {repr(e)}")
                pass
            else:
                output_transcript(o, now=end)

            logger.info(f"## last processed {end:.2f}s\n")

            if end >= duration:
                break
            
            beg = end
            
            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration
    else:  # online = simultaneous mode
        end = 0
        while True:
            now = time.time() - start
            if now < end+min_chunk:
                time.sleep(min_chunk+end-now)
            end = time.time() - start
            logger.info(f"The system received audio from {beg:.2f} s to {end:.2f} s")
            a = load_audio_chunk(audio_path, beg, end)
            beg = end
            online.insert_audio_chunk(a)

            try:
                o = online.process_iter(start_time=start_time)
            except AssertionError as e:
                logger.error(f"assertion error: {e}")
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            logger.info(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}\n")

            if end >= duration:
                break
        now = None

    # Get First Token Latency if available
    first_token_latency = online.first_token_latency
    if first_token_latency is not None:
        print(f"\nFirst Token Latency: {first_token_latency*1000:.2f} ms")
    o = online.finish(start_time=start_time)
    output_transcript(o, now=now)

    # Concatenate all transcriptions into final output
    final_transcription = ""
    if all_transcriptions:
        final_transcription = " ".join([segment['text'] for segment in all_transcriptions])
        print("\n" + "=" * 80)
        print("FINAL TRANSCRIPTION:")
        print(final_transcription)
        print("=" * 80)
    
    return {
        'file': audio_path,
        'duration': duration,
        'segments': all_transcriptions,
        'final_text': final_transcription,
        'first_token_latency': first_token_latency
    }


def main_simulation_from_file(factory, add_args=None):
    '''
    factory: function that creates the ASR and online processor object from args and logger.  
            or in the default WhisperStreaming local agreement backends (not implemented but could be).
    add_args: add specific args for the backend
    '''

    import argparse
    parser = argparse.ArgumentParser()

    processor_args(parser)
    if add_args is not None:
        add_args(parser)

    simulation_args(parser)

    args = parser.parse_args()
    args.offline = False  # TODO: offline mode is not implemented in SimulStreaming yet

    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)

    set_logging(args,logger)

    import os
    audio_path = args.audio_path

    # Check if batch processing is needed
    is_directory = os.path.isdir(audio_path)
    
    if is_directory or args.batch:
        # Batch processing mode
        audio_files = get_audio_files(audio_path, args.audio_extensions)
        
        if not audio_files:
            logger.error(f"No audio files found in: {audio_path}")
            sys.exit(1)
        
        # Limit number of files if specified
        if args.num_audios is not None and args.num_audios > 0:
            audio_files = audio_files[:args.num_audios]
            logger.info(f"Limiting to first {args.num_audios} audio files")
        
        logger.info(f"Found {len(audio_files)} audio files for batch processing")
        
        # Initialize ASR and online processor once
        asr, online = asr_factory(args, factory)
        if args.vac:
            min_chunk = args.vac_chunk_size
        else:
            min_chunk = args.min_chunk_size
        
        # Warm up the ASR with first file
        a = load_audio_chunk(audio_files[0], 0, 1)
        asr.warmup(a)
        
        # Process all files
        batch_results = []
        for idx, audio_file in enumerate(audio_files, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing file {idx}/{len(audio_files)}: {os.path.basename(audio_file)}")
            logger.info(f"{'='*80}")

            try:
                result = process_single_audio_file(audio_file, args, asr, online, min_chunk, factory)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        # Save batch results
        if args.logdir and batch_results:
            os.makedirs(args.logdir, exist_ok=True)
            
            # Calculate average FTL
            ftl_values = [r['first_token_latency'] for r in batch_results if r.get('first_token_latency') is not None]
            avg_ftl = sum(ftl_values) / len(ftl_values) if ftl_values else None
            
            # Save summary JSON
            summary_file = os.path.join(args.logdir, "batch_transcriptions.json")
            summary_data = {
                'total_files': len(audio_files),
                'processed_files': len(batch_results),
                'average_first_token_latency_ms': avg_ftl * 1000 if avg_ftl is not None else None,
                'results': [
                    {
                        'file': os.path.basename(r['file']),
                        'duration': r['duration'],
                        'transcription': r['final_text'],
                        'first_token_latency_ms': r['first_token_latency'] * 1000 if r.get('first_token_latency') else None
                    }
                    for r in batch_results
                ]
            }
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Batch summary saved to: {summary_file}")
            
        
        # Print final summary
        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total files processed: {len(batch_results)}/{len(audio_files)}")
        
        # Print average FTL if available
        if batch_results:
            ftl_values = [r['first_token_latency'] for r in batch_results if r.get('first_token_latency') is not None]
            if ftl_values:
                avg_ftl_ms = (sum(ftl_values) / len(ftl_values)) * 1000
                print(f"Average First Token Latency: {avg_ftl_ms:.2f} ms")
        
        print("=" * 80)
        
        # Run evaluation if reference file is provided
        if args.reference_file and batch_results:
            logger.info(f"\nRunning evaluation against reference file: {args.reference_file}")
            try:
                # Import evaluation module
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from evaluate import load_references, evaluate_transcriptions
                
                # Load references
                references = load_references(args.reference_file, language=args.lan)
                
                # Create generated transcriptions map from batch results
                generated = {}
                ftl_map = {}  # Map filename to FTL
                for result in batch_results:
                    filename = os.path.basename(result['file'])
                    generated[filename] = result['final_text']
                    # Store FTL for this file
                    if result.get('first_token_latency') is not None:
                        ftl_map[filename] = result['first_token_latency']
                
                # Evaluate
                evaluation_results = evaluate_transcriptions(references, generated)
                
                # Add FTL information to evaluation results
                # Calculate average FTL for matched files
                ftl_values = [r['first_token_latency'] for r in batch_results if r.get('first_token_latency') is not None]
                avg_ftl = sum(ftl_values) / len(ftl_values) if ftl_values else None
                evaluation_results['average_first_token_latency_ms'] = avg_ftl * 1000 if avg_ftl is not None else None
                
                # Add per-file FTL to per_file_results
                for result in evaluation_results['per_file_results']:
                    filename = result['file']
                    if filename in ftl_map:
                        result['first_token_latency_ms'] = ftl_map[filename] * 1000
                
                # Print evaluation summary
                print("\n" + "=" * 80)
                print("EVALUATION RESULTS")
                print("=" * 80)
                print(f"Total reference files: {evaluation_results['total_files']}")
                print(f"Matched files: {evaluation_results['matched_files']}")
                print(f"Unmatched files: {evaluation_results['unmatched_files']}")
                print(f"\nAverage CER: {evaluation_results['average_cer']:.4f} ({evaluation_results['average_cer']*100:.2f}%)")
                if evaluation_results.get('average_first_token_latency_ms') is not None:
                    print(f"Average First Token Latency: {evaluation_results['average_first_token_latency_ms']:.2f} ms")
                print("=" * 80)
                
                # Print per-file results if in INFO or DEBUG mode
                if args.log_level in ['DEBUG', 'INFO']:
                    print("\nPer-file CER:")
                    print("-" * 80)
                    for result in evaluation_results['per_file_results']:
                        print(f"{result['file']:40s} CER: {result['cer']:.4f} ({result['cer']*100:.2f}%)")
                
                # Save evaluation results
                eval_output = args.eval_output or os.path.join(args.logdir, 'evaluation_results.json')
                with open(eval_output, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
                logger.info(f"\nEvaluation results saved to: {eval_output}")
                
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                import traceback
                traceback.print_exc()
        
    else:
        # Single file processing mode (original behavior)
        SAMPLING_RATE = 16000
        duration = len(load_audio(audio_path))/SAMPLING_RATE
        logger.info("Audio duration is: %2.2f seconds" % duration)

        asr, online = asr_factory(args, factory)
        if args.vac:
            min_chunk = args.vac_chunk_size
        else:
            min_chunk = args.min_chunk_size

        # load the audio into the LRU cache before we start the timer
        a = load_audio_chunk(audio_path,0,1)

        # warm up the ASR because the very first transcribe takes much more time than the other
        asr.warmup(a)

        # Process the single file
        result = process_single_audio_file(audio_path, args, asr, online, min_chunk, factory)
        
        # Save single file result if logdir is specified
        if args.logdir:
            os.makedirs(args.logdir, exist_ok=True)
            transcript_file = os.path.join(args.logdir, "final_transcription.txt")
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(result['final_text'])
            logger.info(f"Transcription saved to: {transcript_file}")
        
        # Run evaluation if reference file is provided
        if args.reference_file and result['final_text']:
            logger.info(f"\nRunning evaluation against reference file: {args.reference_file}")
            try:
                # Import evaluation module
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from evaluate import load_references, calculate_cer
                
                # Load references
                references = load_references(args.reference_file, language=args.lan)
                
                # Find matching reference
                filename = os.path.basename(audio_path)
                
                ref_text = None
                if filename in references:
                    ref_text = references[filename]
                
                if ref_text:
                    cer = calculate_cer(ref_text, result['final_text'])
                    
                    # Print evaluation result
                    print("\n" + "=" * 80)
                    print("EVALUATION RESULT")
                    print("=" * 80)
                    print(f"File: {filename}")
                    print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
                    print("=" * 80)
                    
                    # Save evaluation result if logdir is specified
                    if args.logdir:
                        eval_output = args.eval_output or os.path.join(args.logdir, 'evaluation_result.json')
                        eval_data = {
                            'file': filename,
                            'reference': ref_text,
                            'generated': result['final_text'],
                            'cer': cer,
                            'ref_length': len(ref_text),
                            'gen_length': len(result['final_text']),
                            'first_token_latency_ms': result['first_token_latency'] * 1000 if result.get('first_token_latency') else None
                        }
                        with open(eval_output, 'w', encoding='utf-8') as f:
                            json.dump(eval_data, f, indent=2, ensure_ascii=False)
                        logger.info(f"Evaluation result saved to: {eval_output}")
                else:
                    logger.warning(f"No reference found for {filename} in reference file")
                    
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                import traceback
                traceback.print_exc()
    
    ######### Evaluation code #########

