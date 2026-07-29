from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type
class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        # Sort corpus by date (newest first) — determines sim column order.
        corpus = sorted(corpus, key=lambda x: x.added_date, reverse=True)

        # Build tag groups from sorted positions. Each group's indices are
        # already in newest-first order since the full corpus is sorted.
        # Papers without tags (or with empty tags) fall into __default__.
        tag_groups: dict[str, list[int]] = {}
        for i, c in enumerate(corpus):
            tags = getattr(c, 'tags', None)
            if not tags:
                tags = ['__default__']
            for tag in tags:
                tag_groups.setdefault(tag, []).append(i)

        candidate_abstracts = [c.abstract for c in candidates]
        corpus_abstracts = [c.abstract for c in corpus]
        sim = self.get_similarity_score(candidate_abstracts, corpus_abstracts)
        assert sim.shape == (len(candidates), len(corpus))

        # Compute per-group scores with independently normalized time-decay
        # so group size doesn't determine influence.
        all_group_scores = []
        for _tag, indices in tag_groups.items():
            group_sim = sim[:, indices]  # columns already newest-first
            time_decay = 1 / (1 + np.log10(np.arange(len(indices)) + 1))
            time_decay = time_decay / time_decay.sum()
            group_score = (group_sim * time_decay).sum(axis=1) * 10
            all_group_scores.append(group_score)

        # Average scores across groups (each group has equal weight).
        scores = np.mean(all_group_scores, axis=0)
        for s, c in zip(scores, candidates):
            c.score = float(s)
        candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        return candidates
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]