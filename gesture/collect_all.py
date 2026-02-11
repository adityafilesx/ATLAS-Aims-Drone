"""
collect_all.py — Interactive Bulk Data Collection
===================================================
Guides the user through collecting data for all 13 gesture classes.
Checks if data already exists and allows skipping.

Usage:
    python -m gesture.collect_all
"""

import os
import sys
import subprocess
from gesture.gesture_labels import GESTURE_LABELS

DATASET_ROOT = "dataset"

def main():
    print("\n" + "=" * 60)
    print("  ATLAS — Interactive Data Collection Guide")
    print("  We will collect 800 training images per class.")
    print("  Press RETURN to start, or Control+C to quit.", flush=True)
    print("=" * 60 + "\n", flush=True)
    
    try:
        input("Press RETURN to begin...")
    except KeyboardInterrupt:
        return
        
    print("\nStarting data collection loop...", flush=True)

    for label in GESTURE_LABELS:
        # Check existing count
        train_dir = os.path.join(DATASET_ROOT, "train", label)
        count = 0
        if os.path.exists(train_dir):
            count = len([f for f in os.listdir(train_dir) if f.endswith(".png")])
        
        print(f"\n--- Class: {label.upper()} ---")
        print(f"  Current images: {count}")
        
        if count >= 800:
            choice = input(f"  Already have {count} images. Collect more? [y/N] ").strip().lower()
            if choice != 'y':
                continue
        else:
            choice = input(f"  Ready to collect '{label}'? [Y/n] ").strip().lower()
            if choice == 'n':
                print("  Skipping...")
                continue

        # Collect Train
        print(f"\n  Starting TRAIN collection for '{label}' (800 images)...")
        print("  → Press 'c' to capture | 'q' to finish.")
        subprocess.run([
            sys.executable, "-m", "gesture.collect_data",
            "--label", label,
            "--split", "train",
            "--count", "800"
        ])

        # Collect Val
        print(f"\n  Starting VAL collection for '{label}' (200 images)...")
        print("  → Press 'c' to capture | 'q' to finish.")
        subprocess.run([
            sys.executable, "-m", "gesture.collect_data",
            "--label", label,
            "--split", "val",
            "--count", "200"
        ])
            
    print("\n" + "=" * 60)
    print("  All done! Now run training:")
    print("  python3 -m gesture.train_model")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
