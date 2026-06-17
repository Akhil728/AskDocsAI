import faiss
import numpy as np
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from app.config import DATA_DIR
from app.services.embedding import EmbeddingService


class VectorStore:
    """FAISS-based vector store for semantic search."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.embedding_dim = embedding_service.get_embedding_dim()
        self.index = None
        self.chunks = []
        self.is_gpu_enabled = False  # Keep simple, GPU caused segfaults

        # Storage paths
        self.vector_store_dir = DATA_DIR / "vector_store"
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.vector_store_dir / "chunks.json"

        # Initialize
        self._initialize_index()
        print(f"🗂️ VectorStore initialized (GPU: {self.is_gpu_enabled}, Dim: {self.embedding_dim})")

    def _create_new_index(self):
        """Create a new CPU FAISS index."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        print("💻 Using CPU for FAISS")

    def _initialize_index(self):
        """Load chunks from disk and rebuild FAISS index from stored embeddings."""
        self._create_new_index()

        if self.chunks_path.exists():
            try:
                with open(self.chunks_path, 'r') as f:
                    self.chunks = json.load(f)

                # Rebuild FAISS from stored embeddings
                chunks_with_embeddings = [c for c in self.chunks if 'embedding' in c]
                if chunks_with_embeddings:
                    embeddings = np.array(
                        [c['embedding'] for c in chunks_with_embeddings],
                        dtype=np.float32
                    )
                    self.index.add(embeddings)

                print(f"📂 Loaded vector store with {len(self.chunks)} chunks")
            except Exception as e:
                print(f"❌ Error loading vector store: {e}")
                self.chunks = []
                self._create_new_index()
        else:
            print("🆕 Creating new FAISS index...")

    def _save_index(self):
        """Save only chunks.json — FAISS is rebuilt from embeddings on load."""
        try:
            with open(self.chunks_path, 'w') as f:
                json.dump(self.chunks, f, indent=2)
            print(f"💾 Saved vector store with {len(self.chunks)} chunks")
        except Exception as e:
            print(f"❌ Error saving vector store: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        print(f"➕ Adding {len(chunks)} chunks to vector store...")

        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_service.generate_embeddings(texts)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.index.add(embeddings.astype(np.float32))

        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            chunk['vector_index'] = start_idx + i
            chunk['embedding'] = embeddings[i].tolist()
            self.chunks.append(chunk)

        print(f"✅ Added {len(chunks)} chunks. Total chunks: {len(self.chunks)}")
        self._save_index()
        return len(chunks)

    def search(self, query: str, top_k: int = 5,
               score_threshold: float = 0.1,
               document_ids: List[str] = None) -> List[Dict[str, Any]]:

        if not self.chunks or self.index.ntotal == 0:
            print("⚠️ Vector store is empty")
            return []

        print(f"🔍 Searching for: '{query[:50]}...' (top_k={top_k}, filter_docs={len(document_ids) if document_ids else 'None'})")

        query_embedding = self.embedding_service.generate_embeddings([f"query: {query}"])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        if document_ids:
            print(f"🎯 Filtering search to {len(document_ids)} selected documents")

            valid_indices = [
                i for i, chunk in enumerate(self.chunks)
                if chunk.get('document_id') in document_ids
            ]

            if not valid_indices:
                print("⚠️ No chunks found for selected documents")
                return []

            print(f"📊 Found {len(valid_indices)} chunks from selected documents")

            subset_embeddings = []
            for idx in valid_indices:
                chunk = self.chunks[idx]
                if 'embedding' in chunk:
                    subset_embeddings.append(np.array(chunk['embedding'], dtype=np.float32))
                else:
                    continue

            if not subset_embeddings:
                print("⚠️ No embeddings found for selected chunks")
                return []

            subset_embeddings = np.array(subset_embeddings, dtype=np.float32)
            subset_index = faiss.IndexFlatIP(self.embedding_dim)
            subset_index.add(subset_embeddings)

            scores, subset_indices = subset_index.search(
                query_embedding.astype(np.float32),
                min(top_k, len(valid_indices))
            )

            results = []
            for score, subset_idx in zip(scores[0], subset_indices[0]):
                if subset_idx >= 0 and score >= score_threshold:
                    original_idx = valid_indices[subset_idx]
                    chunk = self.chunks[original_idx].copy()
                    chunk['similarity_score'] = float(score)
                    results.append(chunk)

            print(f"✅ Returning {len(results)} results from selected documents")

        else:
            scores, indices = self.index.search(
                query_embedding.astype(np.float32),
                min(top_k, len(self.chunks))
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= score_threshold:
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = float(score)
                    results.append(chunk)

            print(f"📊 Found {len(results)} results above threshold {score_threshold}")

        return results

    def remove_by_document_id(self, document_id: str) -> int:
        original_count = len(self.chunks)
        self.chunks = [
            c for c in self.chunks
            if not c.get('document_id', '').startswith(document_id)
        ]
        removed = original_count - len(self.chunks)

        if removed > 0:
            # Rebuild in-memory FAISS index
            self._create_new_index()

            if self.chunks:
                embeddings = np.array(
                    [c['embedding'] for c in self.chunks if 'embedding' in c],
                    dtype=np.float32
                )
                if len(embeddings) > 0:
                    self.index.add(embeddings)

            self._save_index()
            print(f"🗑️ Removed {removed} chunks for document {document_id}")

        return removed

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_chunks': len(self.chunks),
            'embedding_dimension': self.embedding_dim,
            'gpu_enabled': self.is_gpu_enabled,
            'index_size': self.index.ntotal if self.index else 0,
            'model_name': self.embedding_service.model_name
        }

    def clear(self):
        self._create_new_index()
        self.chunks = []
        self._save_index()
        print("🗑️ Vector store cleared")