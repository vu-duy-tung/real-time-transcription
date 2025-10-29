import argparse
import sys
import pandas as pd
import os
import json
import base64
from pathlib import Path
import numpy as _np

#!/usr/bin/env python3
"""
vietmed.py

Simple utility to read a Parquet file and print a short summary.
Default path is:
    /home/duy/PlayWithMino/SimulStreaming/test-data-sample.parquet
"""

def read_parquet(path):
        try:
                return pd.read_parquet(path)
        except Exception as e:
                # fallback to pyarrow if pandas engine isn't available
                try:
                        import pyarrow.parquet as pq
                        table = pq.read_table(path)
                        return table.to_pandas()
                except Exception as e2:
                        raise RuntimeError(f"Failed to read parquet. pandas error: {e}; pyarrow error: {e2}")

def main():
        parser = argparse.ArgumentParser(description="Read a Parquet file and show a brief summary.")
        parser.add_argument("path", nargs="?", default="/home/duy/PlayWithMino/SimulStreaming/test-data-sample.parquet",
                                                help="Path to the parquet file")
        args = parser.parse_args()

        try:
                df = read_parquet(args.path)
        except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)

        # Summary
        print(f"Loaded: {len(df):,} rows, {len(df.columns):,} columns")
        print("\nColumns and dtypes:")
        print(df.dtypes)
        print("\nFirst 10 rows:")
        # limit width to avoid huge output
        # print(df.head(10).to_string(index=False))
        # import code; code.interact(local=locals())
        save_root = Path("/home/duy/PlayWithMino/SimulStreaming/save_dir/data/vietnamese")
        audios_dir = save_root / "audios"
        audios_dir.mkdir(parents=True, exist_ok=True)
        json_path = save_root / "references.json"

        records = []
        for idx, row in df.iterrows():
            audio_obj = row.get("audio") if isinstance(row, pd.Series) else None
            text = row.get("text") if isinstance(row, pd.Series) else None

            if audio_obj is None:
                continue

            # extract bytes and original path if present
            audio_bytes = None
            if isinstance(audio_obj, dict) or hasattr(audio_obj, "get"):
                audio_bytes = audio_obj.get("bytes")
            else:
                audio_bytes = audio_obj

            if audio_bytes is None:
                continue

            # normalize to raw bytes
            try:
                if isinstance(audio_bytes, memoryview):
                    audio_bytes = audio_bytes.tobytes()
                elif isinstance(audio_bytes, bytearray):
                    audio_bytes = bytes(audio_bytes)
                elif isinstance(audio_bytes, bytes):
                    pass
                elif isinstance(audio_bytes, str):
                    # assume base64-encoded string, fall back to utf-8 bytes if decode fails
                    try:
                        audio_bytes = base64.b64decode(audio_bytes)
                    except Exception:
                        audio_bytes = audio_bytes.encode("utf-8")
                else:
                    # list/tuple of ints or numpy array
                    try:
                        if isinstance(audio_bytes, _np.ndarray):
                            audio_bytes = audio_bytes.tobytes()
                        else:
                            audio_bytes = bytes(audio_bytes)
                    except Exception:
                        audio_bytes = bytes(audio_bytes)
            except Exception:
                # skip problematic row
                continue

            safe_idx = str(idx).replace(os.sep, "_")
            out_name = f"{safe_idx}.wav"
            out_path = audios_dir / out_name

            try:
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
            except Exception:
                # skip if can't write file
                continue

            records.append({"audio_path": str(out_path), "index": idx, "text_vi": text})

        # write JSON with utf-8 to preserve Vietnamese characters
        save_root.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(records, jf, ensure_ascii=False, indent=2)

        print(f"Wrote {len(records)} audio files to {audios_dir} and {json_path}")

if __name__ == "__main__":
        main()