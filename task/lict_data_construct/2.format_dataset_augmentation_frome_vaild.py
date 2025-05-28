"""Augmentation for positive. Select pseudo-question from vaild postives."""

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
        return []
    
    valid_chunks = [c for c in filtered_chunks if "案情與心證相關段落" in c["chunk_type"]]
    
    if not valid_chunks:
        logging.warning("No valid pseudo-question found with chunk_type containing '案情與心證相關段落'.")
        return []
    
    result = []
    for pseudo_question_data in valid_chunks:
        """把所有重要的 chunk 當成 pseudo question"""
        pseudo_question = pseudo_question_data["chunk"]
        
        idx = chunks.index(pseudo_question_data)
        start_idx = max(0, idx - window_size)
        end_idx = min(len(chunks), idx + window_size + 1)
        
        pseudo_evidence = [
            c["chunk"] for i, c in enumerate(chunks[start_idx:end_idx]) if i + start_idx != idx
        ]

        result.append((pseudo_question, pseudo_evidence))
        
    return result


def sample_negative_samples(corpus_chunks: list, evidences: list, q: str, neg_num: int):
    """從整個 dataset 的 chunks 欄位中選擇負樣本"""
    evidences = evidences if evidences is not None else []
    possible_negatives = [c["chunk"] for c in corpus_chunks if c["chunk"] != q and c["chunk"] not in evidences]
    
    neg_samples = random.sample(possible_negatives, min(len(possible_negatives), neg_num))
    
    return neg_samples


def process_dataset(dataset, corpus_chunks, pos_num, neg_num):
    processed_data = []
    
    for entry in dataset:
        pseudo_data = extract_pseudo_data(entry['chunks'], pos_num)
        
        if not pseudo_data:
            continue 
        
        for query, pos in pseudo_data:
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
    
    output_dir = f"dataset/lict_ft_data_augmented/{crime_type}"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{crime_type}_{index_type}_{version}.json"
    ut.save_json(output_path, formatted_data)
    logging.info(f"Data save to {output_path}")

    
if __name__ == "__main__":
    main()