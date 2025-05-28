import os
import random
import logging
import argparse

import lib.utils as ut

parser = argparse.ArgumentParser()
parser.add_argument("--crime_type", type=str, required=True, help="Specify the crime type")
parser.add_argument("--version", required=True, help="Choose which version to run")
args = parser.parse_args()

# parameter setting
crime_type = args.crime_type
index_type = "judgment"
version = args.version

pos_num = 1
neg_num = 6

# Configure logging setting
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_pseudo_data(chunks: list, window_size: int):
    filtered_chunks = chunks[1:-1]
    
    if len(filtered_chunks) < 3:
        logging.warning("Chunks list size is less than 3.")
        return None, None
    
    pseudo_question = random.choice(filtered_chunks)

    idx = chunks.index(pseudo_question)
    start_idx = max(0, idx - window_size)
    end_idx = min(len(chunks), idx + window_size + 1)
    
    pseudo_evidence = [
        c for i, c in enumerate(chunks[start_idx:end_idx]) if i + start_idx != idx
    ]
    
    return pseudo_question, pseudo_evidence


def sample_negative_samples(corpus_chunks: list, evidences: list, q: str, neg_num: int):
    """從整個 dataset 的 chunks 欄位中選擇負樣本"""
    evidences = evidences if evidences is not None else []
    possible_negatives = [c for c in corpus_chunks if c != q and c not in evidences]
    
    neg_samples = random.sample(possible_negatives, min(len(possible_negatives), neg_num))
    
    return neg_samples


def process_dataset(dataset, corpus_chunks, pos_num, neg_num):
    processed_data = []
    
    for entry in dataset:
        query, pos = extract_pseudo_data(entry['chunks'], pos_num)
        
        if query is None:
            neg = None
        else:
            neg = sample_negative_samples(corpus_chunks, pos, query, neg_num)
        
        if query and pos and neg:
            processed_data.append({
                "query": query,
                "pos": pos,
                "neg": neg
            })
    
    return processed_data


def main():
    dataset_path = f"dataset/preprocess/lict/label_chunks/{crime_type}/{crime_type}_{index_type}_{version}.json"
    dataset = ut.load_json(dataset_path)

    corpus_chunks = [c for entry in dataset for c in entry['chunks']]

    logging.info(f"原始資料集大小: {len(dataset)}, 語料庫 chunks 數量: {len(corpus_chunks)}")
    
    processed_data = process_dataset(dataset, corpus_chunks, pos_num, neg_num)
    
    logging.info(f"處理完的資料集大小: {len(processed_data)}")
    
    formatted_data = [
        {
            "id": idx,
            "query": data["query"],
            "pos": data["pos"],
            "neg": data["neg"],
            "crime_type": crime_type
        }
        for idx, data in enumerate(processed_data)
    ]
    
    output_dir = f"dataset/lict_ft_data/{crime_type}"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{crime_type}_{index_type}_{version}.json"
    ut.save_json(output_path, formatted_data)
    logging.info(f"Data save to {output_path}")

    
if __name__ == "__main__":
    main()