import re
import random
import logging
import argparse
import configparser
from tqdm import tqdm
import sys
import os

sys.path.append(os.getcwd())


import lib.utils as ut
from lib.CRF import JudgeViewpointClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--crime_type", type=str, required=True, help="Specify the crime type")
args = parser.parse_args()

# parameter setting
crime_type = args.crime_type
mode = 'fact'
neg_num = 5

# Configure logging setting
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load config
config = configparser.ConfigParser()
config.read("config.ini")

embedding_model_name = config["MODEL"]["embedding_model_name"]
classifier_data_path = config["CRF"]["data_path"]
crf_init_data_path = config["CRF"]["init_data_sent_path"]

crf_init_data = ut.load_json(crf_init_data_path)
judge_viewpoints = crf_init_data["judge_viewpoints"]
claims = crf_init_data["claims"]

fact_patterns1 = ['事實一、', '事實壹、', '犯罪事實一、', '犯罪事實壹、']
fact_patterns2 = ['事實及理由一、', '事實及理由壹、', '犯罪事實及理由一、', '犯罪事實及理由壹、']
content_pattern = ['理由一、', '理由壹、', '事實及理由一、', '事實及理由壹、', '犯罪事實及理由一、', '犯罪事實及理由壹、']


def init_classifier():
    logging.info("Initializing classifier...")

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
    
    logging.info("Classifier initialized.")
    return classifier


def chunk_classify(content):
    """
    return: 「案情與心證相關段落」、「無關段落」或「無法分類」
    """
    return classifier.classify_text(content)


def calculate_avg_length(dataset):
    """Calculate the average length of postive, negative, and query."""
    
    pos_lengths = [len(text) for data in dataset for text in data['pos']]
    neg_lengths = [len(text) for data in dataset for text in data['neg']]
    query_lengths = [len(data['query']) for data in dataset]

    avg_pos_length = sum(pos_lengths) / len(pos_lengths) if pos_lengths else 0
    avg_neg_length = sum(neg_lengths) / len(neg_lengths) if neg_lengths else 0
    avg_query_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0

    logging.info("pos 平均長度: {:.2f}".format(avg_pos_length))
    logging.info("neg 平均長度: {:.2f}".format(avg_neg_length))
    logging.info("query 平均長度: {:.2f}".format(avg_query_length))


def extract_fact_content(entry):
    """Extract fact and content from the judgment."""

    judgment = entry['judgment']
    fact, content = 'None', 'None'

    # 提取 fact
    for prefix, pattern in [(prefix, f"{prefix}(.*?)理由一、") for prefix in fact_patterns1] + \
                          [(prefix, f"{prefix}(.*?)書記官") for prefix in fact_patterns2]:
        match = re.search(pattern, judgment, re.DOTALL)
        if match:
            fact = "一、" + match.group(1).strip()
            break

    # 提取 content
    for prefix in content_pattern:
        match = re.search(f"{prefix}(.*?)書記官", judgment, re.DOTALL)
        if match:
            content = "一、" + match.group(1).strip()
            break

    # 合併 fact 和 content
    fact_content = fact if fact == content else f"{fact}\n{content}"

    # 計算剩餘 judgment
    remaining_judgment = judgment.replace(fact, "")
    if mode == 'fact_content':
        remaining_judgment = remaining_judgment.replace(content, "")

    entry.update({
        'fact': fact,
        'content': content,
        'fact_content': fact_content,
        'remaining_judgment': remaining_judgment
    })

    return entry


def format_ds(dataset):
    logging.info("Formatting dataset...")

    ds = [
        extract_fact_content({
            'no': entry['no'],
            'reason': entry['reason'],
            'judgment': entry['judgment'], 
        })
        for entry in tqdm(dataset, desc="Processing dataset", unit="entry")
    ]
    
    logging.info("Dataset formatted.")
    return ds


def extract_ds(dataset):
    logging.info("Extracting dataset...")

    process_data = [
        {
            'no': entry['no'],
            'reason': entry['reason'],
            'judgment': entry['judgment'],
            'fact': entry['fact'],
            'remaining_judgment': entry['remaining_judgment'],
        }
        for entry in tqdm(dataset, desc="Extracting dataset", unit="entry")
    ]
    
    output_path = f"dataset/preprocess/lcaet/sentence_level/extract/{crime_type}_judgment.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ut.save_json(output_path, process_data)

    logging.info(f"Dataset extracted and saved to {output_path}.")
    return process_data


def label_chunks(dataset):
    logging.info("Labeling chunks...")

    for entry in tqdm(dataset, desc="Labeling chunks", unit="entry"):
        entry['chunks'] = []
        for chunk in ut.split_into_sentences(entry['fact']):
            chunk_type = chunk_classify(chunk)
            entry['chunks'].append({
                'chunk': chunk,
                'chunk_type': chunk_type
            })
    
    output_path = f"dataset/preprocess/lcaet/sentence_level/label_chunks/{crime_type}_judgments.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ut.save_json(output_path, dataset)

    logging.info(f"Labeled chunks saved to {output_path}.")

    return dataset


def construct_data_w_neg(dataset):
    logging.info("Construct data...")

    format_ds = [
        {
            'no': entry['no'],
            'crime_type': entry['reason'],
            'query': entry['judgment'],
            'pos': ["\n".join(chunk["chunk"] for chunk in entry["chunks"] if "案情與心證相關段落" in chunk["chunk_type"])],
            'neg': ["\n".join(chunk["chunk"] for chunk in entry["chunks"] if "案情與心證相關段落" not in chunk["chunk_type"])]

        }   
        for entry in dataset
    ]

    calculate_avg_length(format_ds)

    logging.info("Construct complete.")

    return format_ds


def construct_data_w_more_neg(format_ds, neg_num):
    logging.info("Sample negtive samples...")
    # 確保每個元素的 neg 都是列表
    for entry in format_ds:
        entry['neg'] = list(entry['neg'])  # 確保是可變列表

    for idx, entry in enumerate(format_ds):
        other_negs = [item['neg'][0] for i, item in enumerate(format_ds) if i != idx]  # 排除自己
        sampled_negs = random.sample(other_negs, min(neg_num, len(other_negs)))  # 隨機取 5 個（若不足 5 個則全取）
        entry['neg'].extend(sampled_negs)  # 加入 neg 陣列

    calculate_avg_length(format_ds)

    output_path = f"dataset/lcaet_ft_data/sentence_level/{crime_type}_judgments.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ut.save_json(output_path, format_ds)
    
    logging.info(f"Sampled data saved to {output_path}")

def main():
    global classifier
    logging.info("Starting main process...")
    classifier = init_classifier()

    dataset_path = f"dataset/judgments/{crime_type}_judgments.json"
    logging.info(f"Loading dataset from {dataset_path}...")
    dataset = ut.load_json(dataset_path)
    
    ds = format_ds(dataset[:3])
    process_ds = extract_ds(ds)
    label_ds = label_chunks(process_ds)

    construct_ds = construct_data_w_neg(label_ds)
    construct_data_w_more_neg(construct_ds, neg_num=neg_num)

    logging.info("Processing completed.")

if __name__ == "__main__":
    main()