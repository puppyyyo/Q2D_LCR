import os
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.getcwd())

import lib.utils as ut
from lib.get_gpt import init_openai_client, get_gpt_response, parse_json_response

parser = argparse.ArgumentParser(description="Data collection for judgment highlight task.")
parser.add_argument("--crime_type", type=str, required=True, help="Crime type for dataset selection.")
args = parser.parse_args()

crime_type = args.crime_type

client = init_openai_client()

with open("task/llm_replace_CSA/extract_sentences.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()


def main():
    ds = ut.load_json(f"dataset/preprocess/llm_replace_csa/format/{crime_type}_judgments.json")
    # print(ds[0])

    for entry in tqdm(ds):
        fact_content = entry['fact_content']

        prompt = prompt_template.replace("{fact_content}", fact_content)
        response = get_gpt_response(client, prompt)
        response_dict = parse_json_response(response)
        saliency_sentences = response_dict.get("saliency_sentences", [])

        # 把每句最後一個字切掉 (通常是句號)
        truncated_sentences = [s[:-1] for s in saliency_sentences]

        # 比對是否在 judgment 中
        valid_sentences = [s for s in truncated_sentences if s in fact_content]

        entry['response'] = response
        entry['saliency_sentences'] = truncated_sentences
        entry['valid_sentences'] = valid_sentences

        print(f"number of saliency sentences: {len(saliency_sentences)}")
        print(f"number of valid sentences: {len(valid_sentences)}")

        result = {
            "id": entry['id'],
            "no": entry['no'],
            "court": entry['court'],
            "reason": entry['reason'],
            "mainText": entry['mainText'],
            "judgment": entry['judgment'],
            "fact_content": entry['fact_content'],
            "remaining_judgment": entry['remaining_judgment'],
            "response": entry['response'],
            "saliency_sentences": entry['saliency_sentences'],
            "valid_sentences": entry['valid_sentences']
        }

        ut.save_jsonl(f"dataset/preprocess/llm_replace_csa/llm_label/{crime_type}.jsonl", result)

if __name__ == '__main__':
    main()