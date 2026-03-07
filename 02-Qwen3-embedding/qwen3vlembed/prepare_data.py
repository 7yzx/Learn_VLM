import os
import json
import pandas as pd
import requests
import glob
import random
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Configuration
PARQUET_GLOB = 'datasets/conceptual_captions/labeled/*.parquet'
IMAGE_DIR = 'dataset/images'
OUTPUT_TRAIN_JSONL = 'dataset/train.jsonl'
OUTPUT_EVAL_JSON = 'dataset/eval.json'
MAX_SAMPLES = 100000  # Number of samples
EVAL_RATIO = 0.1
CANDIDATE_MULTIPLIER = 3
MAX_ASPECT_RATIO = 200.0
SEED = 42
NUM_WORKERS = 256

def download_and_process_image(image_url, caption, idx):
    """
    Downloads an image and returns the formatted entry if successful.
    """
    try:
        response = requests.get(image_url, timeout=5)
        if response.status_code != 200:
            return None
            
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB') # Ensure RGB

        width, height = img.size
        if width <= 0 or height <= 0:
            return None
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > MAX_ASPECT_RATIO:
            return None
        
        image_filename = f"image_{idx}.jpg"
        image_path = os.path.abspath(os.path.join(IMAGE_DIR, image_filename))
        img.save(image_path)
        
        # Multimodal Format: (Image+Text) <-> (Text)
        # Query: Image + "Find a description for this image."
        # Positive: Caption (text-only)
        
        entry = {
            "messages": [
                {
                    "role": "user", 
                    "content": "<image> Find a description for this image."
                }
            ],
            "images": [image_path],
            "positive_messages": [
                [
                    {
                        "role": "user", 
                        "content": caption
                    }
                ]
            ]
        }
        return entry
        
    except Exception:
        return None

def prepare_data():
    # 1. Create directories
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    
    parquet_files = sorted(glob.glob(PARQUET_GLOB))
    if not parquet_files:
        print(f"Error: No parquet files found with pattern: {PARQUET_GLOB}")
        return

    print(f"Reading parquet files: {len(parquet_files)}")
    dfs = []
    for fp in parquet_files:
        try:
            dfs.append(pd.read_parquet(fp, columns=["image_url", "caption"]))
        except Exception as e:
            print(f"Warning: failed to read {fp}: {e}")
    if not dfs:
        print("Error: failed to read any parquet files.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    if df_all.empty:
        print("Error: combined dataframe is empty.")
        return

    random.seed(SEED)
    n_candidates = min(len(df_all), MAX_SAMPLES * CANDIDATE_MULTIPLIER)
    df_subset = df_all.sample(n=n_candidates, random_state=SEED).reset_index(drop=True)
    
    print(f"Processing {len(df_subset)} candidates to get {MAX_SAMPLES} valid samples...")
    
    data_items = []
    
    # 3. Download and process in parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for idx, row in enumerate(df_subset.itertuples(index=False), start=0):
            futures.append(executor.submit(download_and_process_image, row.image_url, row.caption, idx))
            
        with tqdm(total=len(futures)) as pbar:
            for future in futures:
                result = future.result()
                if result:
                    data_items.append(result)
                    if len(data_items) >= MAX_SAMPLES:
                        # Cancel remaining if possible or just stop adding
                        break
                pbar.update(1)

    eval_size = max(1, int(len(data_items) * EVAL_RATIO))
    train_items = data_items[:-eval_size] if eval_size < len(data_items) else []
    eval_items = data_items[-eval_size:] if eval_size < len(data_items) else data_items

    print(f"Saving {len(train_items)} train samples to {OUTPUT_TRAIN_JSONL}...")
    os.makedirs(os.path.dirname(OUTPUT_TRAIN_JSONL), exist_ok=True)
    with open(OUTPUT_TRAIN_JSONL, 'w', encoding='utf-8') as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saving {len(eval_items)} eval samples to {OUTPUT_EVAL_JSON}...")
    os.makedirs(os.path.dirname(OUTPUT_EVAL_JSON), exist_ok=True)
    with open(OUTPUT_EVAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(eval_items, f, ensure_ascii=False, indent=2)

    print(f"Done! Train: {OUTPUT_TRAIN_JSONL}, Eval: {OUTPUT_EVAL_JSON}")

if __name__ == "__main__":
    prepare_data()
