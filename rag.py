import glob
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICY_DIR = "data/policies"
FALLBACK_PREFIX = "I could not find"


def chunk_documents(size=40, overlap=10, folder=POLICY_DIR):
    """Word chunks of `size` words, stepping by size-overlap; stop after the chunk that reaches the last word."""
    step, chunks = size - overlap, []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            chunks.append({"source": os.path.basename(path), "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += step
    return chunks


class Retriever:
    def __init__(self, size=40, overlap=10):
        self.chunks = chunk_documents(size, overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in order]

    def answer(self, question, k=3):
        hits = self.retrieve(question, k)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [
            {"role": "system", "content": "You are a POLICY ASSISTANT for Instamart support. Answer only from the supplied "
                                          "context. If the context does not contain the answer, say you could not find that "
                                          "information in the policy documents."},
            {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
        ]
        resp = mock_llm.chat(messages)
        return resp.choices[0].message.content, [h["source"] for h in hits]


_default = None


def default():
    global _default
    if _default is None:
        _default = Retriever()
    return _default


def retrieve(query, k=3):
    return default().retrieve(query, k)


def answer(question):
    return default().answer(question)
