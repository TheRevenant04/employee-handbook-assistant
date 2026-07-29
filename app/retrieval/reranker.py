from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


MAX_CONTENT_CHARS = 1000


class Reranker:
    def __init__(self, path="models/Xenova/ms-marco-MiniLM-L-6-v2", max_length=512, use_sigmoid=False):
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path / "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_names = {inp.name for inp in self.session.get_inputs()}
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.max_length = max_length
        self.use_sigmoid = use_sigmoid

        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise ValueError("Tokenizer is missing [PAD] token")
        self.pad_id = pad_id

        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(
            pad_id=self.pad_id,
            pad_type_id=0,
            pad_token="[PAD]",
        )

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def score(self, query, documents):
        if not documents:
            return []

        contents = []
        for doc in documents:
            text = doc.get("content", "") if isinstance(doc, dict) else doc
            contents.append(text[:MAX_CONTENT_CHARS])

        encoded = self.tokenizer.encode_batch([(query, c) for c in contents])

        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.asarray([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.asarray([e.attention_mask for e in encoded], dtype=np.int64)
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.asarray([e.type_ids for e in encoded], dtype=np.int64)

        missing = self.input_names - set(feed.keys())
        if missing:
            raise ValueError(f"Missing required model inputs: {sorted(missing)}")

        outputs = self.session.run(self.output_names or None, feed)
        logits = np.asarray(outputs[0])

        if logits.ndim == 2 and logits.shape[1] == 1:
            scores = logits[:, 0]
        elif logits.ndim == 1:
            scores = logits
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

        if self.use_sigmoid:
            scores = self._sigmoid(scores)

        return scores.astype(float).tolist()

    def rerank(self, query, documents, top_k=None):
        if not documents:
            return []

        scores = self.score(query, documents)

        scored_docs = []
        for doc, score in zip(documents, scores):
            if isinstance(doc, dict):
                item = {**doc, "rerank_score": score}
            else:
                item = {"content": doc, "rerank_score": score}
            scored_docs.append(item)

        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            scored_docs = scored_docs[:top_k]

        return scored_docs