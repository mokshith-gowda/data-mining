import os
import glob
import re
import pandas as pd
from datasketch import MinHash, MinHashLSH

NUM_PERM = 128
LSH_THRESHOLD = 0.63

def clean_and_shingle(text):
    if not isinstance(text, str):
        return set()
    text = re.sub(r'\b(?:\$|Rs\.?|INR)\s*\d+(?:,\d+)*(?:\.\d+)?\b', '<MONEY>', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '<DATE>', text)
    text = re.sub(r'[A-Z0-9]{6,}', '<REF>', text)
    tokens = re.findall(r'\w+', text.lower())
    return set(" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2))

def run_pipeline():
    print("=== STARTING SETUBID DEDUPLICATION PIPELINE ===")
    
    if os.path.exists('exam/data/labeled_pairs.csv'):
        labels_df = pd.read_csv('exam/data/labeled_pairs.csv')
        print("\n--- Step 1: Label Skew Analysis ---")
        print(labels_df['label'].value_counts(normalize=True))
    
    parquet_files = glob.glob('exam/data/notices/*.parquet')
    if not parquet_files:
        print("No parquet files found.")
        return
    
    notices_df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    print(f"\n--- Loaded {len(notices_df)} notices ---")
    
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    for idx, row in notices_df.iterrows():
        notice_id = str(row.get('notice_id', idx))
        text = str(row.get('title', '')) + " " + str(row.get('body', ''))
        shingles = clean_and_shingle(text)
        m = MinHash(num_perm=NUM_PERM)
        for s in shingles:
            m.update(s.encode('utf-8'))
        lsh.insert(notice_id, m)
        
    print("LSH Index successfully built for all notices.")

if __name__ == '__main__':
    run_pipeline()