import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# Configuration
DATASET_ID = "AIMLOps-C4-G16/IndianFestivals"
OUTPUT_DIR = Path("data")
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Mapping labels to rich prompts for Stable Diffusion
LABEL_MAP = {
    'Christmas': "Christmas celebration, decorated Christmas tree, colorful lights, festive decorations, Santa Claus, gifts, holiday season",
    'Diwali': "Diwali festival of lights, clay oil diyas, fireworks, rangoli patterns, Hindu celebration India, golden illumination",
    'EID': "Eid celebration, mosque architecture, crescent moon, Muslim festival, prayers, traditional attire, lanterns",
    'Ganesh Chaturthi': "Ganesh Chaturthi festival, Lord Ganesha idol, colorful decorations, Hindu celebration, traditional worship India",
    'Holi': "Holi festival of colors, people throwing colored powder, spring celebration, vibrant colors India, joyful crowd",
    'Independence Day': "India Independence Day celebration, national flag, patriotic display, parade, saffron white green colors",
    'Lohri': "Lohri festival celebration, bonfire, Punjabi harvest festival, traditional dance, winter celebration, peanuts and popcorn"
}

PROMPT_SUFFIX = ", vibrant photography, high quality, cultural festival celebration"

def main():
    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, trust_remote_code=True)
    
    # The dataset usually comes as a single 'train' split
    full_ds = ds['train']
    print(f"Total samples: {len(full_ds)}")
    
    # Split the dataset
    train_testval = full_ds.train_test_split(test_size=0.2, seed=42)
    test_val = train_testval['test'].train_test_split(test_size=0.5, seed=42)
    
    splits = {
        "train": train_testval['train'],
        "val": test_val['train'],
        "test": test_val['test']
    }
    
    # Process each split
    label_names = full_ds.features['label'].names
    
    for split_name, split_data in splits.items():
        print(f"Processing {split_name} split ({len(split_data)} samples)...")
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        for i, sample in enumerate(tqdm(split_data)):
            img = sample['image']
            label_idx = sample['label']
            label_name = label_names[label_idx]
            
            # Generate file path
            file_name = f"{label_name.replace(' ', '_').lower()}_{i:05d}.png"
            file_path = split_dir / file_name
            
            # Save image
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(file_path)
            
            # Map label to prompt
            base_prompt = LABEL_MAP.get(label_name, f"{label_name} festival celebration")
            rich_prompt = base_prompt + PROMPT_SUFFIX
            
            # Append to metadata
            metadata.append({
                "file_name": file_name,
                "text": rich_prompt,
                "label": label_name
            })
            
        # Write metadata.jsonl
        metadata_path = split_dir / "metadata.jsonl"
        with open(metadata_path, 'w') as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")
                
    print(f"Success! Data saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
