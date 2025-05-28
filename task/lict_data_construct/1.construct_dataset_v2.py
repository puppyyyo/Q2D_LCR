import os
import configparser
import logging
import argparse
from tqdm import tqdm

import lib.utils as ut
from lib.CRF import JudgeViewpointClassifier

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

# Check config.ini exists
if not os.path.exists("config.ini"):
    raise FileNotFoundError("Missing config.ini file. Please create one before running the script.")

# Load config
config = configparser.ConfigParser()
config.read("config.ini")

embedding_model_name = config["MODEL"]["embedding_model_name"]
classifier_data_path = f"dataset/judgments/{crime_type}_judgments.json"
crf_init_data_path = config["CRF"]["init_data_path"]

crf_init_data = ut.load_json(crf_init_data_path)
judge_viewpoints = crf_init_data["judge_viewpoints"]
claims = crf_init_data["claims"]

def init_classifier():
    classifier = JudgeViewpointClassifier(
        model_name=embedding_model_name,
        use_fp16=True,
        num_clusters=8,
        random_state=42,
        data_file=classifier_data_path,
        max_samples=400,
        min_length=5,
        separators=r"。|；|：|，",
        judge_viewpoints=judge_viewpoints,
        claims=claims
    )

    return classifier

def chunk_classify(content):
    """
    return: 「案情與心證相關段落」、「無關段落」或「無法分類」
    """
    return classifier.classify_text(content)
    
def process_chunks(dataset):
    for idx, entry in enumerate(tqdm(dataset, desc="Processing entries")):
        logging.info(f"Process ID {idx}...")
        entry['chunks'] = []  # 初始化為空列表

        for chunk in ut.split_into_sentences(entry['judgment']):
            chunk_type = chunk_classify(chunk)
            entry['chunks'].append({"chunk": chunk, "chunk_type": chunk_type})

            logging.info(f"Chunk: {chunk}")
            logging.info(f"Chunk type: {chunk_type}")

    output_dir = f"dataset/preprocess/lict/label_chunks/{crime_type}"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{crime_type}_{index_type}_{version}.json"
    ut.save_json(output_path, dataset)
    logging.info(f"Data save to {output_path}")


def main():
    global classifier
    classifier = init_classifier()

    dataset_path = f"dataset/judgments/{crime_type}_judgments.json"
    dataset = ut.load_json(dataset_path)

    process_chunks(dataset)

if __name__ == "__main__":
    main()