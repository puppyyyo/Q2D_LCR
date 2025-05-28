import re
import random
import logging
import argparse
import sys
import os

sys.path.append(os.getcwd())


import lib.utils as ut

parser = argparse.ArgumentParser(description="Data collection for judgment highlight task.")
parser.add_argument("--crime_type", type=str, required=True, help="Crime type for dataset selection.")
parser.add_argument("--neg_num", type=int, default=5, help="Number of negative samples.")
args = parser.parse_args()

crime_type = args.crime_type

fact_patterns1 = ['事實一、', '事實壹、', '犯罪事實一、', '犯罪事實壹、']
fact_patterns2 = ['事實及理由一、', '事實及理由壹、', '犯罪事實及理由一、', '犯罪事實及理由壹、']
content_pattern = ['理由一、', '理由壹、', '事實及理由一、', '事實及理由壹、', '犯罪事實及理由一、', '犯罪事實及理由壹、']

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    remaining_judgment = remaining_judgment.replace(content, "")

    entry.update({
        'fact': fact,
        'content': content,
        'fact_content': fact_content,
        'remaining_judgment': remaining_judgment
    })

    return entry


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


def format_dataset(dataset):
    """Extract fact and content from the judgment."""
    logging.info("Extracting fact and content from the judgment...")

    ds = [
        extract_fact_content({
            'no': entry['no'],
            'reason': entry['reason'],
            'judgment': entry['judgment'],
        })
        for entry in dataset
    ]

    output_path = f"dataset/preprocess/lcaet/paragraph_level/{crime_type}_judgment.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ut.save_json(output_path, ds)

    logging.info("Format complete.")
    return ds



def filter_dataset(dataset):
    """Filter dataset by removing fact or content equal to 'None'."""
    logging.info("Filtering dataset...")

    removed = [entry['no'] for entry in dataset if entry['fact'] == 'None' or entry['content'] == 'None']
    logging.info(f"Removed {len(removed)} entries: {removed}")

    filtered_ds = [entry for entry in dataset if entry['fact'] != 'None' and entry['content'] != 'None']

    logging.info("Filtering complete.")
    return filtered_ds



def construct_data_w_neg(ds):
    """Construct data with negative sample from the remaining judgment."""
    logging.info("Constructing data with negative samples...")

    format_data = [
        {
            'no': entry['no'],
            'id': idx,
            'crime_type': entry['reason'],
            'query': entry['judgment'],
            'pos': [entry['fact_content']],
            'neg': [entry['remaining_judgment']]
        }
        for idx, entry in enumerate(ds)
    ]

    calculate_avg_length(format_data)

    logging.info("Data construction with negative samples complete.")

    return format_data


def construct_data_w_more_neg(format_data, neg_num):
    """Construct data with negative sample from other entries."""
    logging.info("Constructing data with more negative samples...")

    for entry in format_data:
        entry['neg'] = list(entry['neg'])

    for idx, entry in enumerate(format_data):
        other_negs = [item['neg'][0] for i, item in enumerate(format_data) if i != idx]
        sampled_negs = random.sample(other_negs, min(neg_num, len(other_negs)))
        entry['neg'].extend(sampled_negs)

    calculate_avg_length(format_data)

    output_path = f"dataset/lcaet_ft_data/paragraph_level/{crime_type}_judgment.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ut.save_json(output_path, format_data)

    logging.info(f"Data size is {len(format_data)}.\nData is save to {output_path}")
    logging.info("Data construction with more negative samples complete.")


def main():
    """Main function to handle dataset preparation."""
    logging.info("Starting the fine-tuning data preparation process...")

    dataset = ut.load_json(f"dataset/judgments/{crime_type}_judgments.json")
    logging.info(f"Loaded dataset with {len(dataset)} entries.")

    raw_ds = format_dataset(dataset)
    filter_ds = filter_dataset(raw_ds)

    format_data = construct_data_w_neg(filter_ds)

    construct_data_w_more_neg(format_data, args.neg_num)
    logging.info("Data preparation complete.")


if __name__ == '__main__':
    main()