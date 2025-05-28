import re
import json


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj_list: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as wf:
        json.dump(obj_list, wf, ensure_ascii=False)


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")


def split_into_chunks(
    text: str,
    chunk_size: int,
    overlap_size: int,
    separator: str = r"。[一二三四五六七八九十]、|[一二三四五六七八九十]、"
) -> list:
    
    # 若沒有指定 separator，直接按照 chunk_size、overlap_size 切分
    if not separator:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap_size)]
    
    # 若指定了 separator（可支援 regex），先行切分文字
    splitted_texts = re.split(separator, text)
    
    # 若 chunk_size == 0，直接返回切分結果
    if chunk_size == 0:
        return [part.strip() for part in splitted_texts if part.strip()]
    
    chunks = []
    for part in splitted_texts:
        # 去除前後空白，避免空字串
        part = part.strip()
        if not part:
            continue
        
        # 將切出來的每個部分，再依 chunk_size、overlap_size 繼續切分
        start = 0
        while start < len(part):
            end = start + chunk_size
            chunks.append(part[start:end])
            start += (chunk_size - overlap_size)

    return chunks

def split_into_sentences(text: str):
    return [sentence.strip() for sentence in text.split("。") if sentence.strip()]

def extract_fact_content(entry):
    crimes_list1 = ['事實一、', '事實壹、', '犯罪事實一、', '犯罪事實壹、']
    crimes_list2 = ['事實及理由一、', '事實及理由壹、', '犯罪事實及理由一、', '犯罪事實及理由壹、']
    crimes_list3 = ['理由一、', '理由壹、', '事實及理由一、', '事實及理由壹、', '犯罪事實及理由一、', '犯罪事實及理由壹、']

    judgment = entry['judgment']
    fact, content = 'None', 'None'

    # 提取 fact
    for crime in crimes_list1:
        if crime in judgment:
            pattern = f"{crime}(.*?)理由一、"
            match = re.search(pattern, judgment, re.DOTALL)
            if match:
                fact = "一、" + match.group(1).strip()
                break
    for crime in crimes_list2:
        if crime in judgment:
            pattern = f"{crime}(.*?)書記官"
            match = re.search(pattern, judgment, re.DOTALL)
            if match:
                fact = "一、" + match.group(1).strip()
                break

    # 提取 content
    for crime in crimes_list3:
        if crime in judgment:
            pattern = f"{crime}(.*?)書記官"
            match = re.search(pattern, judgment, re.DOTALL)
            if match:
                content = "一、" + match.group(1).strip()
                break

    # 合併 fact_content
    fact_content = fact if fact == content else f"{fact}\n{content}"

    entry.update({
        'fact': fact,
        'content': content,
        'fact_content': fact_content,
        'remaining_judgment': judgment.replace(fact, "").replace(content, "")
    })
    return entry
