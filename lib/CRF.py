# 引入必要的庫
import re
import os
import json
import configparser
from collections import Counter

from sklearn.cluster import KMeans
from sklearn_crfsuite import CRF, metrics

from FlagEmbedding import BGEM3FlagModel

# Check config.ini exists
if not os.path.exists("config.ini"):
    raise FileNotFoundError("Missing config.ini file. Please create one before running the script.")

# Load config
config = configparser.ConfigParser()
config.read("config.ini")

embedding_model_name = config["MODEL"]["embedding_model_name"]

class JudgeViewpointClassifier:
    def __init__(self, 
                 model_name='BAAI/bge-m3', 
                 use_fp16=True, 
                 num_clusters=3, 
                 random_state=42,
                 data_file='larceny.json',
                 max_samples=400,
                 min_length=5,
                 separators=r"。|；|：|，",
                 judge_viewpoints=None,
                 claims=None):
        """
        初始化 JudgeViewpointClassifier 類別，並準備CRF的訓練資料。

        :param model_name: 使用的嵌入模型名稱
        :param use_fp16: 是否使用16位浮點數
        :param num_clusters: 聚類數量
        :param random_state: 隨機種子
        :param data_file: JSON 文件路徑
        :param max_samples: 最多處理的樣本數量
        :param min_length: 保留的最小句子長度
        :param separators: 分句符號
        :param judge_viewpoints: 法官觀點相關文字列表
        :param claims: 事實相關文字列表
        """
        self.model = self.initialize_model(model_name, use_fp16)
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.kmeans = None
        self.crf = None

        # 加載並處理數據
        self.thoughts_pool = self.load_and_process_data(
            file_path=data_file,
            max_samples=max_samples,
            min_length=min_length,
            separators=separators
        )

        # 聚類文本
        self.cluster_texts(self.thoughts_pool)

        # 定義法官觀點和事實相關文字
        if judge_viewpoints is None:
            self.judge_viewpoints = [
                "核被告所為，",
                "核被告鄭明全所",
            ]
        else:
            self.judge_viewpoints = judge_viewpoints

        if claims is None:
            self.claims = [    
                "乙○○與成年女子甲○（真實名籍詳卷）係於公園晨運相識。",
                "陳嘉麟明知其女友A女（代號為0000-000000，民國89年4月情",
            ]
        else:
            self.claims = claims

        # 預測聚類標籤
        self.judge_viewpoints_clusters = self.predict_clusters(self.judge_viewpoints)
        self.claims_clusters = self.predict_clusters(self.claims)

        # 將聚類標籤轉換為列表
        self.judge_viewpoints_cluster_sequences = [seq.tolist() for seq in self.judge_viewpoints_clusters] 
        self.claims_cluster_sequences = [seq.tolist() for seq in self.claims_clusters]

        # 準備訓練數據
        X_A, y_A = self.prepare_data(self.judge_viewpoints_cluster_sequences, 'A')
        X_B, y_B = self.prepare_data(self.claims_cluster_sequences, 'B')

        # 合併數據
        X = X_A + X_B
        y = y_A + y_B

        # 訓練 CRF 模型
        self.train_crf_model(X, y)

    def initialize_model(self, model_name=embedding_model_name, use_fp16=True):
        """
        初始化嵌入模型。

        :param model_name: 模型名稱
        :param use_fp16: 是否使用16位浮點數
        :return: 初始化的嵌入模型
        """
        return BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def load_and_process_data(self, file_path, max_samples=400, min_length=5, separators=r"。|；|：|，"):
        """
        讀取並處理數據。

        :param file_path: JSON 文件路徑
        :param max_samples: 最多處理的樣本數量
        :param min_length: 保留的最小句子長度
        :param separators: 分句符號
        :return: 處理後的句子列表
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取判決文字
        thoughts = [d['judgment'] for d in data[:max_samples]]
        
        # 分句
        thoughts_pool = [sentence.strip() for thought in thoughts for sentence in re.split(separators, thought)]
        
        # 清理空句子和短句子
        thoughts_pool = [thought for thought in thoughts_pool if thought and len(thought) > min_length]
        
        print(f"總句子數量: {len(thoughts_pool)}")
        return thoughts_pool

    def cluster_texts(self, thoughts_pool):
        """
        聚類文本。

        :param thoughts_pool: 處理後的句子列表
        :return: 訓練好的 KMeans 模型和聚類標籤
        """
        # 將文本轉換為嵌入向量
        embeddings = self.model.encode(thoughts_pool)['dense_vecs']
        
        # 初始化並訓練KMeans模型
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        self.kmeans.fit(embeddings)
        
        # 獲取聚類標籤
        cluster_labels = self.kmeans.labels_
        
        # 打印每個聚類的一些示例
        for i in range(self.num_clusters):
            print(f"\nCluster {i}:")
            for j, thought in enumerate(thoughts_pool):
                if cluster_labels[j] == i and j % 100 == 0:
                    print(f"- {thought}")
            print('-' * 30)
        
        return self.kmeans, cluster_labels

    def predict_clusters(self, sequences, separators=r"。|；|：|，"):
        """
        預測序列的聚類標籤。

        :param sequences: 文本序列列表
        :param separators: 分句符號
        :return: 聚類標籤序列列表
        """
        cluster_sequences = []
        for seq in sequences:
            # 分句
            segments = [sentence.strip() for sentence in re.split(separators, seq) if sentence.strip()]
            print(f"句子數量: {len(segments)}")
            
            if len(segments) == 0:
                cluster_sequences.append([])
                continue
            
            # 將句子轉換為嵌入向量
            embeddings = self.model.encode(segments)['dense_vecs']
            
            # 預測聚類
            predicted_clusters = self.kmeans.predict(embeddings)
            cluster_sequences.append(predicted_clusters)
        
        return cluster_sequences

    @staticmethod
    def extract_features(sequence, index):
        """
        為序列中的每個元素提取特徵。

        :param sequence: 聚類標籤序列
        :param index: 當前索引
        :return: 特徵字典
        """
        element = sequence[index]
        features = {
            'current': element,
            'is_first': index == 0,
            'is_last': index == len(sequence) - 1,
            'prev': sequence[index - 1] if index > 0 else '<START>',
            'next': sequence[index + 1] if index < len(sequence) - 1 else '<END>',
        }
        return features

    def prepare_data(self, sequences, label):
        """
        準備訓練數據。

        :param sequences: 聚類標籤序列列表
        :param label: 標籤（例如 'A' 或 'B'）
        :return: 特徵數據和標籤數據
        """
        X = []
        y = []
        for seq in sequences:
            if not seq:
                continue  # 跳過空序列
            X_seq = [self.extract_features(seq, i) for i in range(len(seq))]
            y_seq = [label] * len(seq)  # 對整個序列分配相同的標籤
            X.append(X_seq)
            y.append(y_seq)
        return X, y

    def train_crf_model(self, X, y):
        """
        訓練 CRF 模型。

        :param X: 特徵數據
        :param y: 標籤數據
        :return: 訓練好的 CRF 模型
        """
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,  # L1 正則化系數
            c2=0.1,  # L2 正則化系數
            max_iterations=100,
            all_possible_transitions=True
        )
        self.crf.fit(X, y)
        print("CRF 模型訓練完成。")
        return self.crf

    def classify_sequence(self, sequence):
        """
        使用 CRF 模型對單個序列進行分類。

        :param sequence: 聚類標籤序列
        :return: 預測的標籤
        """

        X_test = [self.extract_features(sequence, i) for i in range(len(sequence))]
        y_pred = self.crf.predict([X_test])[0]
        
        # 取出出現頻率最高的標籤作為整個序列的類別
        label_counts = Counter(y_pred)
        predicted_label = label_counts.most_common(1)[0][0]
        return predicted_label

    def classify_text(self, text):
        """
        對單個文本進行分類，判斷其屬於「案情與心證相關段落」還是「非相關段落」。

        :param text: 要分類的文本
        :return: 分類結果
        """
        # 預測聚類
        cluster_sequence = self.predict_clusters([text])[0]
        
        # 使用 CRF 模型進行分類
        predicted_label = self.classify_sequence(cluster_sequence)
        
        if predicted_label == 'A':
            return "判定此序列屬於 案情與心證相關段落"
        elif predicted_label == 'B':
            return "判定此序列屬於 無關段落"
        else:
            return "無法判定此序列的類別"

    def evaluate_model(self, X_test, y_test):
        """
        評估 CRF 模型的性能。

        :param X_test: 測試特徵數據
        :param y_test: 測試標籤數據
        :return: 評估報告
        """
        y_pred = self.crf.predict(X_test)
        return metrics.flat_classification_report(y_test, y_pred)
