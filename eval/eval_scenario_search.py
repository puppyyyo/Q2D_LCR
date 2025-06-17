import gc
import torch
import faiss
import numpy as np
from tqdm import tqdm
from FlagEmbedding import FlagModel
from datasets import load_dataset
from FlagEmbedding.abc.evaluation.utils import evaluate_metrics

crime_type = "larceny"
model_type = "m3"
split = "subset"

def instance_model(model_name):
    return FlagModel(
        model_name_or_path=model_name,
        use_fp16=False
    )


def search(model, queries, corpus, top_k):
    queries_text = queries["text"]
    corpus_text = corpus["text"]

    queries_embeddings = model.encode_queries(queries_text)
    corpus_embeddings = model.encode_corpus(corpus_text)

    dim = corpus_embeddings.shape[-1]
    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    index.train(corpus_embeddings)
    index.add(corpus_embeddings)

    all_scores, all_indices = [], []
    for i in tqdm(range(0, len(queries_embeddings), 32), desc="Searching"):
        j = min(i + 32, len(queries_embeddings))
        batch_queries = queries_embeddings[i: j].astype(np.float32)
        scores, indices = index.search(batch_queries, k=top_k)
        all_scores.append(scores)
        all_indices.append(indices)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)

    results = {}
    for idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
        query_id = queries["id"][idx]
        results[query_id] = {}
        for score, index in zip(scores, indices):
            if index != -1:
                corpus_id = corpus["id"][index]
                results[query_id][corpus_id] = float(score)

    return results


def main():
    k_values = [1, 5, 10, 20, 50]
    top_k = 100

    version_list = [
        "BAAI/bge-m3",
        f"./models/lict/data_num_ablation/{crime_type}-{model_type}-lict_v2_50",
        f"./models/lict/data_num_ablation/{crime_type}-{model_type}-lict_v2_100",
        f"./models/lict/data_num_ablation/{crime_type}-{model_type}-lict_v2_200",
        f"./models/lict/data_num_ablation/{crime_type}-{model_type}-lict_v2_300",
        f"./models/lict/data_num_ablation/{crime_type}-{model_type}-lict_v2_400",
        f"./models/lict/data_num_ablation/{crime_type}-{model_type}-lict_v2_500",
        f"./models/lict/data_num_ablation/{crime_type}-{model_type}-lict_v2_600",
        f"puppyyyo/{crime_type}-{model_type}-subset-ICT_v2"
    ]

    dataset_path = f"dataset/eval_data_syn_500/{crime_type}/format"
    corpus = load_dataset("json", data_files=f"{dataset_path}/corpus.json")["train"]
    queries = load_dataset("json", data_files=f"{dataset_path}/queries.json")["train"]
    qrels = load_dataset("json", data_files=f"{dataset_path}/qrels.json")["train"]
    qrels_dict = {str(q["qid"]): {str(q["docid"]): q["relevance"]} for q in qrels}

    for model_name in version_list:
        print(f"Loading model: {model_name}")
        model = instance_model(model_name)

        results = search(model, queries, corpus, top_k)

        results_str = {
            str(qid): {str(docid): score for docid, score in docs.items()}
            for qid, docs in results.items()
        }

        ndcg, map_, recall, precision = evaluate_metrics(qrels_dict, results_str, k_values)
        print(f"Recall@K: {recall}")

        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
