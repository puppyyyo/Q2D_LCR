"""
Script: construct_chunks.py

Program:
    Process legal judgments by splitting into chunks, optionally classifying each chunk using a CRF-based classifier (CSA).

Usage:
    python construct_chunks.py --crime_type <crime_type> --version <v1|v2>

Arguments:
    --crime_type : (required) Specify the crime type, e.g., "larceny", "forgery", "snatch", "fraud" etc.
    --version    : (required) Dataset version.
                   v1: Process chunks without classification.
                   v2: Process chunks with CRF-based classification (CSA).

Example:
    python construct_chunks.py --crime_type larceny --use_CSA
"""

import os
import sys
import logging
import argparse
import configparser
from tqdm import tqdm

sys.path.append(os.getcwd())

import lib.utils as ut

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--crime_type", type=str, required=True, help="Specify the crime type")
parser.add_argument("--version", required=True, choices=["v1", "v2"], help="Dataset version: v1 (no CSA), v2 (with CSA)")
args = parser.parse_args()

# Parameter settings
crime_type = args.crime_type
version = args.version
use_CSA = version == "v2"

# Optional classifier components
classifier = None
judge_viewpoints, claims = [], []

# Configure logging
logger = logging.getLogger(__name__)

log_dir = "logs/lict_ft_data"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"{crime_type}_judgment_{version}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def init_classifier():
    """Initialize CRF-based classifier with config and templates."""
    global judge_viewpoints, claims

    try:
        from lib.CRF import JudgeViewpointClassifier
    except ImportError:
        raise ImportError("Missing lib/CRF.py or dependencies.")

    if not os.path.exists("config.ini"):
        raise FileNotFoundError("Missing config.ini file.")

    config = configparser.ConfigParser()
    config.read("config.ini")

    embedding_model_name = config["MODEL"]["embedding_model_name"]
    crf_init_data_path = config["CRF"]["init_data_path"]
    crf_init_data = ut.load_json(crf_init_data_path)

    judge_viewpoints = crf_init_data["judge_viewpoints"]
    claims = crf_init_data["claims"]

    return JudgeViewpointClassifier(
        model_name=embedding_model_name,
        use_fp16=True,
        num_clusters=8,
        random_state=42,
        data_file=f"dataset/judgments/full/{crime_type}_judgments.json",
        max_samples=400,
        min_length=5,
        separators=r"。|；|：|，",
        judge_viewpoints=judge_viewpoints,
        claims=claims
    )

def chunk_classify(content):
    return classifier.classify_text(content)

def process_chunks(dataset):
    """Process each judgment into chunks, optionally label with CSA."""
    for idx, entry in enumerate(tqdm(dataset, desc="Processing entries")):
        logging.info(f"Processing ID {idx}...")
        entry['chunks'] = []

        chunks = ut.split_into_chunks(entry['judgment'], chunk_size=0, overlap_size=0)
        for chunk in chunks:
            if use_CSA:
                chunk_type = chunk_classify(chunk)
                entry['chunks'].append({"chunk": chunk, "chunk_type": chunk_type})
                logging.debug(f"Chunk: {chunk} -> Type: {chunk_type}")
            else:
                entry['chunks'].append(chunk)
        
    output_dir = f"dataset/preprocess/lict/label_chunks/{crime_type}"
    os.makedirs(output_dir, exist_ok=True)

    version_tag = "v2" if use_CSA else "v1"
    output_path = f"{output_dir}/{crime_type}_judgment_{version_tag}.json"
    ut.save_json(output_path, dataset)
    logging.info(f"Data saved to {output_path}")

def main():
    if use_CSA:
        global classifier
        classifier = init_classifier()

    dataset_path = f"dataset/judgments/subset/{crime_type}_judgments.json"
    dataset = ut.load_json(dataset_path)

    process_chunks(dataset)

if __name__ == "__main__":
    main()
