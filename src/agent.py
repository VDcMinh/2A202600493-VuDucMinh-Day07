from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved_chunks = self.store.search(question, top_k=top_k)
        if retrieved_chunks:
            context = "\n\n".join(
                (
                    f"[Chunk {index} | score={chunk['score']:.3f} | "
                    f"doc_id={chunk['metadata'].get('doc_id', 'unknown')}]"
                    f"\n{chunk['content']}"
                )
                for index, chunk in enumerate(retrieved_chunks, start=1)
            )
        else:
            context = "No relevant context was retrieved from the knowledge base."

        prompt = (
            "You are a retrieval-augmented assistant. Use only the context below to answer "
            "the user's question. If the context is insufficient, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return str(self.llm_fn(prompt))
