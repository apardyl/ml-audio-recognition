import faiss
import torch


# noinspection PyArgumentList
class Searcher:
    def __init__(self, index: faiss.Index):
        self.index = index

    def save(self, file_path: str):
        faiss.write_index(self.index, file_path)

    def search(self, x: torch.Tensor, k: int = 1):
        x = x.cpu().numpy()
        return self.index.search(x, k)

    def add(self, x: torch.Tensor):
        x = x.cpu().numpy()
        self.index.add(x)

    @classmethod
    def load(cls, file_path: str):
        index = faiss.read_index(file_path)
        return cls(index=index)

    @classmethod
    def get_simple_index(cls, embedding_dim):
        return cls(index=faiss.IndexFlatL2(embedding_dim))
