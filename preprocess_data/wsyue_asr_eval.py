import argparse
import csv
import json
import os
import re
import sys
from typing import List, Dict, Optional

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process a tab-separated metadata txt into a JSON list with:
- audio_path
- index
- text_yue

Input lines are expected like:
<filename.wav>\t<age>\t<sentiment>\t<gender>\t<text>

Usage:
    python wsyue_asr_eval.py \
            --input /path/to/metadata.txt \
            --output /path/to/output.json \
            --audio-dir /path/to/audio/root \
            --index-source line|filename \
            --index-start 0
"""



def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
        p = argparse.ArgumentParser(description="Convert WSYUE ASR txt metadata to JSON.")
        p.add_argument("--input", "-i", help="Path to the input txt file (UTF-8, TSV).", default="/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/content.txt")
        p.add_argument("--output", "-o", help="Path to the output json file.", default="/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/content.json")
        p.add_argument("--audio-dir", "-a", help="Base directory to prefix to audio filename.", default="/home/duy/PlayWithMino/SimulStreaming/save_dir/data/WSYue-ASR-eval/Short/wav_")
        p.add_argument(
                "--index-source",
                choices=["line", "filename"],
                default="line",
                help="How to derive 'index': sequential line number or digits parsed from filename stem.",
        )
        p.add_argument(
                "--index-start",
                type=int,
                default=0,
                help="Starting value for line-based indexing (only used when --index-source=line).",
        )
        return p.parse_args(argv)


_digit_re = re.compile(r"(\d+)")


def index_from_filename(name: str) -> Optional[int]:
        """
        Extract the first integer found in the filename stem (without extension).
        Returns None if no digits found.
        """
        stem = os.path.splitext(os.path.basename(name))[0]
        m = _digit_re.search(stem)
        return int(m.group(1)) if m else None


def build_entry(
        audio_filename: str,
        text: str,
        idx_mode: str,
        seq_index: int,
        audio_dir: str,
) -> Dict[str, object]:
        if idx_mode == "filename":
                idx = index_from_filename(audio_filename)
                if idx is None:
                        # Fallback to sequential if no digits found
                        idx = seq_index
        else:
                idx = seq_index

        audio_path = os.path.join(audio_dir, audio_filename) if audio_dir else audio_filename

        return {
                "audio_path": audio_path,
                "index": idx,
                "text_yue": text,
        }


def convert(input_path: str, output_path: str, audio_dir: str, index_source: str, index_start: int) -> None:
        entries: List[Dict[str, object]] = []
        total = 0
        skipped = 0

        with open(input_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                seq_idx = index_start
                for row in reader:
                        total += 1
                        # Skip empty or comment lines
                        if not row or (len(row) == 1 and not row[0].strip()):
                                skipped += 1
                                continue
                        if row[0].strip().startswith("#"):
                                skipped += 1
                                continue

                        # Expect at least [filename, ..., text]; take first and last column robustly
                        if len(row) < 2:
                                skipped += 1
                                continue

                        audio_filename = row[0].strip()
                        text = row[-1].strip()

                        if not audio_filename or not text:
                                skipped += 1
                                continue

                        entry = build_entry(
                                audio_filename=audio_filename,
                                text=text,
                                idx_mode=index_source,
                                seq_index=seq_idx,
                                audio_dir=audio_dir,
                        )
                        entries.append(entry)
                        seq_idx += 1

        # Write JSON
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(entries, out_f, ensure_ascii=False, indent=2)

        print(f"Converted data saved to: {output_path}")
        # Minimal stderr summary
        print(f"Processed: {len(entries)} entries; skipped: {skipped}; total lines: {total}", file=sys.stderr)


def main():
        args = parse_args()
        convert(
                input_path=args.input,
                output_path=args.output,
                audio_dir=args.audio_dir,
                index_source=args.index_source,
                index_start=args.index_start,
        )


if __name__ == "__main__":
        main()