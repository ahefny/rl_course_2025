import numpy as np
import torch
from transformers import BertModel, BertTokenizer

_BERT_EMBEDDER_CACHE: dict[str, tuple[BertModel, BertTokenizer]] = {}

class TextEmbedder:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
    
    def embed(self, text: str) -> torch.Tensor:
        raise NotImplementedError

def _get_bert_embedder(model_name: str = "google-bert/bert-base-uncased", device: str = "cpu") -> tuple[BertModel, BertTokenizer]:
    if model_name not in _BERT_EMBEDDER_CACHE:
        model = BertModel.from_pretrained(model_name, device_map=device)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model.eval()
        _BERT_EMBEDDER_CACHE[model_name] = (model, tokenizer)
    return _BERT_EMBEDDER_CACHE[model_name]

class BertEmbedder(TextEmbedder):
    def __init__(self, model_name: str = "google-bert/bert-base-uncased", embedding_dim: int = 128, device: str = "cpu"):
        super().__init__(embedding_dim=embedding_dim)
        self.model, self.tokenizer = _get_bert_embedder(model_name, device)
        self._projection_matrix = np.random.randn(embedding_dim, self.model.config.hidden_size).astype(np.float32)
        self._projection_matrix = torch.from_numpy(self._projection_matrix).to(device)
        self.device = device


    @torch.inference_mode()
    def embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        embedding = outputs.pooler_output.squeeze(0)
        embedding = self._projection_matrix @ embedding
        norm = embedding.norm()
        if norm > 0:
            embedding = embedding / norm
        else:
            embedding = torch.zeros_like(embedding)
        return embedding