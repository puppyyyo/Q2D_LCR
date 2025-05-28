# LICT + LCAET 微調模型 Scenario search embedding model

資料集包含 larceny, forgery, snatch, fraud


# 流程
## 安裝套件
```bash
pip install -r requirements.txt
```

## LICT
### Usage

```bash
python process_chunks.py \
    --crime_type <CRIME_TYPE> \
    --version <VERSION> \
    --chunk_method <METHOD> \
    [--use_classifier]
```
### Arguments
- `crime_type` (str): 指定要處理的犯罪類型，例如 larceny, forgery, snatch, fraud
- `version` (str): 設定版本名稱，例如 wo_CSA, w_CSA
- `chunk_method` (str): 
    - `sentence`: 用句號切割句子
    - `bullet`: 用列點切割句子
- `use_classifier` (flag): 設定是否要用 CSA


| Version | Chunk Method | Query Method |
| - | - | - |
| v1 | Judgment 照「句號」切 | 隨機抽 |
| v2 | Judgment 照「句號」切 | 從「事實與心證相關」抽 |
| v4 | Judgment 照「列點」切 | 隨機抽 |
| v5 | Judgment 照「列點」切 | 從「事實與心證相關」抽 |