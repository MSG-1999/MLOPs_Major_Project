import os
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# Constants
SOURCE_DIR = Path("data/train")
TARGET_BASE = Path("data_mini")
IMAGES_PER_CLASS = 100
SPLIT_RATIO = {"train": 80, "val": 10, "test": 10}
SEED = 42

def prepare_mini_dataset():
    random.seed(SEED)
    
    metadata_path = SOURCE_DIR / "metadata.jsonl"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found.")
        return

    # 1. Read and group metadata by label
    class_data = defaultdict(list)
    with open(metadata_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            class_data[item['label']].append(item)

    print(f"Found {len(class_data)} classes.")

    # 2. Select and split data
    final_splits = {"train": [], "val": [], "test": []}
    
    for label, items in class_data.items():
        if len(items) < IMAGES_PER_CLASS:
            print(f"Warning: Class {label} only has {len(items)} images. Taking all.")
            selected = items
        else:
            selected = random.sample(items, IMAGES_PER_CLASS)
            
        random.shuffle(selected)
        
        # Calculate split indices
        train_end = SPLIT_RATIO["train"]
        val_end = train_end + SPLIT_RATIO["val"]
        
        final_splits["train"].extend(selected[:train_end])
        final_splits["val"].extend(selected[train_end:val_end])
        final_splits["test"].extend(selected[val_end:])
        
        print(f"Processed class: {label}")

    # 3. Create directories and copy/link data
    for split, items in final_splits.items():
        split_dir = TARGET_BASE / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        new_metadata_path = split_dir / "metadata.jsonl"
        with open(new_metadata_path, 'w') as f_out:
            for item in items:
                # Original filename
                src_file = SOURCE_DIR / item['file_name']
                dst_file = split_dir / item['file_name']
                
                # Symlink image (faster and saves space)
                if dst_file.exists():
                    dst_file.unlink()
                os.symlink(src_file.absolute(), dst_file)
                
                # Write to new metadata file
                f_out.write(json.dumps(item) + '\n')
                
        print(f"Finished {split} split with {len(items)} images.")

if __name__ == "__main__":
    prepare_mini_dataset()
