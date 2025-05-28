import os
import sys
import argparse
import logging

sys.path.append(os.getcwd())

import lib.utils as ut

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["lecard", 'judgments'], required=True, help="Dataset name.")
parser.add_argument("--crime_type", type=str, choices=["larceny", "forgery", "snatch", "fraud"], help="Required if dataset is 'judgments'. Specify the crime type.")
args = parser.parse_args()

# Parameter settings

def load_dataset(dataset_name: str):
    if dataset_name == 'lecard':
        path = "dataset/lecard_v2/corpus_700.json"
    elif dataset_name == 'judgments':
        if not args.crime_type:
            raise ValueError("When using 'judgments' dataset, you must specify --crime_type.")
        path = f"dataset/judgments/{args.crime_type}_judgments.json"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    logging.info(f"Loading dataset from: {path}")
    data = ut.load_json(path)
    logging.info(f"Number of entries: {len(data)}")

    if data:
        logging.info(f"First entry (preview):\n{json_preview(data[0])}")
    else:
        logging.warning("Dataset is empty!")

    return data

def json_preview(obj, max_chars=500):
    import json
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")

if __name__ == "__main__":
    dataset = load_dataset(args.dataset)
