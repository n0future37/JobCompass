"""固定本地模型；新job环境运行，完全离线加载模型资产。"""

import threading

import numpy as np

from .settings import Settings, project_path


class LocalModels:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.lock = threading.RLock()
        self.embed_tokenizer = self.embed_model = None
        self.rank_tokenizer = self.rank_model = None

    def _load(self, kind):
        import torch
        from transformers import (
            AutoModel,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = self.settings.config[kind]
        path = project_path(config["path"])
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        cls = AutoModel if kind == "embedding" else AutoModelForSequenceClassification
        model = cls.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        return tokenizer, model.to(self.device).eval()

    def encode(self, texts, query=False):
        with self.lock:
            if self.embed_model is None:
                self.embed_tokenizer, self.embed_model = self._load("embedding")
            config = self.settings.config["embedding"]
            rows = []
            if query:
                texts = ["为这个句子生成表示以用于检索相关文章：" + text for text in texts]
            for start in range(0, len(texts), config["batch_size"]):
                tokens = self.embed_tokenizer(
                    texts[start : start + config["batch_size"]],
                    padding=True,
                    truncation=True,
                    max_length=config["max_length"],
                    return_tensors="pt",
                ).to(self.device)
                with self.torch.inference_mode():
                    vectors = self.embed_model(**tokens).last_hidden_state[:, 0]
                    vectors = self.torch.nn.functional.normalize(vectors.float(), p=2, dim=1)
                    rows.extend(vectors.cpu().numpy())
            return np.asarray(rows, dtype=np.float32)

    def rerank(self, query, jobs):
        return self.score_passages(query, [j["title"] + "\n" + j["jd_text"] for j in jobs])

    def score_passages(self, query, passages):
        with self.lock:
            if self.rank_model is None:
                self.rank_tokenizer, self.rank_model = self._load("reranker")
            config = self.settings.config["reranker"]
            scores = []
            for start in range(0, len(passages), config["batch_size"]):
                pairs = [[query, passage] for passage in passages[start : start + config["batch_size"]]]
                tokens = self.rank_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=config["max_length"],
                    return_tensors="pt",
                ).to(self.device)
                with self.torch.inference_mode():
                    values = self.rank_model(**tokens).logits.view(-1).float().cpu().tolist()
                    scores.extend(values)
            return scores
