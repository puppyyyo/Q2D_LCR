import os
import logging
import argparse
from tqdm import tqdm

import lib.utils as ut

parser = argparse.ArgumentParser()
parser.add_argument("--crime_type", type=str, required=True, help="Specify the crime type")
parser.add_argument("--version", required=True, help="Choose which version to run")
args = parser.parse_args()

# parameter setting
crime_type = args.crime_type
index_type = "judgment"
version = args.version

# Configure logging setting
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
def process_chunks(dataset):
    for idx, entry in enumerate(tqdm(dataset, desc="Processing entries")):
        logging.info(f"Process ID {idx}...")
        entry['chunks'] = ut.split_into_chunks(entry['judgment'], chunk_size=0, overlap_size=0)

    output_dir = f"dataset/preprocess/lict/label_chunks/{crime_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/{crime_type}_{index_type}_{version}.json"
    ut.save_json(output_path, dataset)
    logging.info(f"Data save to {output_path}")


def main():
    dataset_path = f"dataset/judgments/{crime_type}_judgments.json"
    dataset = ut.load_json(dataset_path)

    process_chunks(dataset)

if __name__ == "__main__":
    main()