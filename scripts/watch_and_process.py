#!/usr/bin/env python3
"""Watch ~/Documents for new MP4 files and process them with demo_video_streaming.py."""

import os
import time
import subprocess
import sys
from pathlib import Path
from glob import glob

WATCH_DIR = "/mnt/HardDisk/hmr/"
SCRIPT = "scripts/demo_video_streaming.py"
POLL_INTERVAL = 2  # seconds
PROCESSED_FILE = os.path.join(WATCH_DIR, ".promptHMR_processed.txt")


def load_processed():
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE) as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def mark_processed(path):
    with open(PROCESSED_FILE, "a") as f:
        f.write(path + "\n")

def get_mp4_files():
    return set(glob(os.path.join(WATCH_DIR, "*.mp4")))

def main():
    print(f"Watching {WATCH_DIR} for new MP4 files...")
    print(f"Processed log: {PROCESSED_FILE}")
    print(f"Press Ctrl+C to stop.\n")
    
    processed = load_processed()
    # Mark existing MP4s as processed on first run

    print("\nWaiting for new files...\n")

    while True:
        try:
            current = get_mp4_files()
            new_files = [f for f in current if f not in processed]
            
            if new_files:
                new_files.sort(key=lambda x: os.path.getmtime(x))
                for mp4_path in new_files:
                    print(f"\n{'='*60}")
                    print(f"New file detected: {mp4_path}")
                    print(f"Processing...")
                    print(f"{'='*60}\n")
                    
                    cmd = [
                        sys.executable, SCRIPT,
                        "--input_video", str(mp4_path)
                    ]
                    
                    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
                    
                    if result.returncode != 0:
                        print(f"\n⚠️  Processing failed for {mp4_path} (exit code {result.returncode})")
                    
                    mark_processed(mp4_path)
                    processed.add(mp4_path)
                    print(f"\n✅ Done with {mp4_path}\n")   
                    print(f"{'='*60}\n")
            
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\nStopped.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
