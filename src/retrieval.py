"""
Advanced hybrid retrieval system with FAISS vectorstore,
multi-query generation, and specialty clustering.
"""
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any, Optional, Tuple
import structlog
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass, field
import asyncio
from pathlib import Path
import os

logger = structlog.get_logger()

def _get_default_llm_model() -> str:
    """Get default LLM model based on API configuration."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "")

    # Check if using Perplexity
    if api_key.startswith("pplx-") or "perplexity" in api_base.lower():
        return os.getenv("LLM_MODEL", "sonar-pro")
    else:
        return os.getenv("LLM_MODEL", "gpt-4o-mini")

@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    openai_model: str = field(default_factory=_get_default_llm_model)
    embedding_model: str = "text-embedding-3-small"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_queries: int = 3
    retrieval_k: int = 10
    final_k: int = 5
    cluster_k: int = 5
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    use_local_embeddings: bool = True  # Use free local embeddings by default

class HybridRetriever:
    """Advanced hybrid retrieval with FAISS vectorstore."""

    def __init__(self, config: RetrievalConfig = RetrievalConfig()):
        self.config = config
        self.llm = ChatOpenAI(model=config.openai_model, temperature=0)

        # Use local embeddings if configured (free, no API calls)
        self._local_embedder = config.use_local_embeddings
        if self._local_embedder:
            logger.info("Using local HuggingFace embeddings (free, no API)")
            self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        else:
            logger.info("Using OpenAI embeddings (paid API)")
            self.embeddings = OpenAIEmbeddings(model=config.embedding_model)

        self.cross_encoder = CrossEncoder(config.cross_encoder_model)

        # FAISS setup
        self.vectorstore = None
        self.documents = []
        self.clusterer = None

    def initialize_vectorstore(self, documents: List[Dict[str, Any]], force_rebuild: bool = False):
        """Initialize FAISS vector store with documents.

        Args:
            documents: List of document dictionaries
            force_rebuild: If True, rebuild index even if it exists. Default: False
        """
        faiss_path = "./faiss_index"

        # Try to load existing index first (unless force_rebuild is True)
        if not force_rebuild and Path(faiss_path).exists():
            logger.info(f"Found existing FAISS index at {faiss_path}")
            if self.load_vectorstore(faiss_path):
                logger.info("Loaded existing FAISS index - skipping rebuild!")

                # Still need to prepare documents for BM25 retrieval
                self.documents = []
                for doc in documents:
                    self.documents.append(
                        Document(page_content=doc["text"], metadata=doc["metadata"])
                    )

                logger.info("Vector store ready (loaded from cache) [OK]")
                return
            else:
                logger.warning("Failed to load existing index, will rebuild...")

        # Build new index
        logger.info(f"Building new FAISS index with {len(documents)} documents")

        # Convert to LangChain documents
        self.documents = []
        for doc in documents:
            self.documents.append(
                Document(page_content=doc["text"], metadata=doc["metadata"])
            )

        # Create FAISS vectorstore
        logger.info("Creating FAISS index with embeddings (this takes ~5-6 minutes)...")
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)

        # Save to disk
        self.vectorstore.save_local(faiss_path)
        logger.info(f"FAISS index saved to {faiss_path}")

        # Generate embeddings for clustering
        logger.info("Generating embeddings for clustering (this takes ~4-7 minutes)...")
        texts = [doc.page_content for doc in self.documents]
        embeddings_list = self.embeddings.embed_documents(texts)

        # Initialize clustering
        metadatas = [doc.metadata for doc in self.documents]
        self._initialize_clustering(embeddings_list, metadatas)

        logger.info("Vector store initialized successfully [OK]")

    def load_vectorstore(self, path: str = "./faiss_index"):
        """Load existing FAISS vectorstore from disk."""
        try:
            logger.info(f"Loading FAISS index from {path}")
            self.vectorstore = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded successfully [OK]")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def _initialize_clustering(self, embeddings: List[List[float]], metadatas: List[Dict]):
        """Initialize KMeans clustering for specialty grouping."""
        if len(embeddings) < self.config.cluster_k:
            logger.warning("Not enough documents for clustering")
            return

        embeddings_array = np.array(embeddings)
        self.clusterer = KMeans(n_clusters=self.config.cluster_k, random_state=42)
        cluster_labels = self.clusterer.fit_predict(embeddings_array)

        # Analyze cluster specialties
        cluster_specialties = {}
        for i, label in enumerate(cluster_labels):
            specialty = metadatas[i].get("specialty", "general")
            if label not in cluster_specialties:
                cluster_specialties[label] = {}
            cluster_specialties[label][specialty] = cluster_specialties[label].get(specialty, 0) + 1

        # Assign dominant specialty to each cluster
        self.cluster_info = {}
        for cluster_id, specialties in cluster_specialties.items():
            dominant_specialty = max(specialties.items(), key=lambda x: x[1])[0]
            self.cluster_info[cluster_id] = {
                "dominant_specialty": dominant_specialty,
                "document_indices": [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            }

        logger.info(f"Initialized clustering with {self.config.cluster_k} specialty clusters [OK]")

    def _multi_query_generate(self, query: str) -> List[str]:
        """Generate multiple query variants using LLM."""
        prompt = f"""You are a medical coding assistant. Generate {self.config.max_queries} different
        variations of the following medical query to improve retrieval of relevant ICD-10-CM and CPT codes:

        Original query: {query}

        Generate queries that:
        1. Use medical synonyms and terminology
        2. Focus on different aspects (symptoms, procedures, diagnoses)
        3. Include common abbreviations or alternate phrasings

        Return only the queries, one per line, without numbering."""

        try:
            response = self.llm.invoke(prompt)
            variants = [line.strip() for line in response.content.split('\n') if line.strip()]
            return variants[:self.config.max_queries]
        except Exception as e:
            logger.error(f"Failed to generate query variants: {e}")
            return [query]

    def _semantic_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Perform semantic search using FAISS."""
        try:
            # FAISS similarity_search_with_score returns (doc, distance)
            # where distance is L2 distance (lower is better)
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            documents_with_scores = []
            for doc, distance in results:
                # Convert L2 distance to similarity score (0-1, higher is better)
                # Using exponential decay: score = exp(-distance)
                score = np.exp(-distance)
                documents_with_scores.append((doc, score))

            return documents_with_scores
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _bm25_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Perform BM25 keyword search."""
        try:
            bm25_retriever = BM25Retriever.from_documents(
                documents=self.documents,
                k=k
            )
            results = bm25_retriever.invoke(query)

            # BM25Retriever doesn't return scores, so we estimate them
            documents_with_scores = []
            for i, doc in enumerate(results):
                # Simple scoring based on term frequency
                score = self._calculate_bm25_score(query, doc.page_content)
                documents_with_scores.append((doc, score))

            return sorted(documents_with_scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _calculate_bm25_score(self, query: str, text: str) -> float:
        """Simple BM25-like scoring for compatibility."""
        query_terms = query.lower().split()
        text_lower = text.lower()

        score = 0.0
        for term in query_terms:
            if term in text_lower:
                tf = text_lower.count(term)
                score += tf * 0.1  # Simplified scoring

        return min(score, 1.0)  # Normalize to [0, 1]

    def _cross_encoder_rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        try:
            scores = self.cross_encoder.predict(pairs)

            # Combine documents with scores
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            reranked_docs = [doc for doc, score in doc_scores[:top_k]]

            logger.info(f"Cross-encoder reranked {len(documents)} to top {len(reranked_docs)} [OK]")
            return reranked_docs

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return documents[:top_k]

    def _filter_by_specialty(self, documents: List[Document], target_specialty: str = None) -> List[Document]:
        """Filter documents by medical specialty using clustering."""
        if not target_specialty or not self.clusterer:
            return documents

        filtered_docs = []
        for doc in documents:
            doc_specialty = doc.metadata.get("specialty", "general")
            if doc_specialty.lower() == target_specialty.lower():
                filtered_docs.append(doc)

        # If no exact matches, use clustering to find related specialties
        if not filtered_docs and len(documents) > 0:
            # Find cluster with target specialty
            target_cluster = None
            for cluster_id, info in self.cluster_info.items():
                if info["dominant_specialty"].lower() == target_specialty.lower():
                    target_cluster = cluster_id
                    break

            if target_cluster is not None:
                cluster_doc_indices = self.cluster_info[target_cluster]["document_indices"]
                for doc in documents:
                    # This is simplified - in practice, you'd maintain document-to-index mapping
                    if doc.metadata.get("specialty", "general").lower() == target_specialty.lower():
                        filtered_docs.append(doc)

        return filtered_docs if filtered_docs else documents

    def retrieve(self, query: str, specialty_filter: str = None) -> List[Document]:
        """Main retrieval method with all enhancements."""
        logger.info(f"Retrieving for query: '{query}' (specialty: {specialty_filter})")

        if not self.vectorstore:
            logger.error("Vectorstore not initialized")
            return []

        # Stage 1: Multi-query generation
        llm_queries = self._multi_query_generate(query)

        # Combine original query with variants
        all_queries = [query] + llm_queries
        logger.info(f"Generated {len(all_queries)} query variants")

        # Stage 2: Hybrid retrieval for each query
        all_semantic_results = []
        all_bm25_results = []

        for q in all_queries:
            semantic_results = self._semantic_search(q, k=self.config.retrieval_k)
            bm25_results = self._bm25_search(q, k=self.config.retrieval_k)

            all_semantic_results.extend(semantic_results)
            all_bm25_results.extend(bm25_results)

        # Stage 3: Fusion and deduplication
        # Combine semantic and BM25 results with weighted scoring
        doc_scores = {}

        # Process semantic results
        for doc, score in all_semantic_results:
            doc_key = doc.page_content
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {"doc": doc, "semantic": 0, "bm25": 0}
            doc_scores[doc_key]["semantic"] = max(doc_scores[doc_key]["semantic"], score)

        # Process BM25 results
        for doc, score in all_bm25_results:
            doc_key = doc.page_content
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {"doc": doc, "semantic": 0, "bm25": 0}
            doc_scores[doc_key]["bm25"] = max(doc_scores[doc_key]["bm25"], score)

        # Calculate combined scores
        scored_documents = []
        for doc_key, scores in doc_scores.items():
            combined_score = (
                scores["semantic"] * self.config.semantic_weight +
                scores["bm25"] * self.config.bm25_weight
            )
            scored_documents.append((scores["doc"], combined_score))

        # Sort by combined score
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        candidate_docs = [doc for doc, score in scored_documents[:self.config.retrieval_k]]

        # Stage 4: Specialty filtering
        if specialty_filter:
            candidate_docs = self._filter_by_specialty(candidate_docs, specialty_filter)

        # Stage 5: Cross-encoder reranking
        final_docs = self._cross_encoder_rerank(query, candidate_docs, self.config.final_k)

        logger.info(f"Retrieved {len(final_docs)} documents after full pipeline")
        return final_docs

def main():
    """Demo retrieval system."""
    from data_loader import MedicalDataLoader

    # Load sample data
    loader = MedicalDataLoader()
    icd_codes = loader.parse_icd10_codes(Path("data/icd10_processed.csv"))
    cpt_codes = loader.parse_cpt_codes(Path("data/cpt_processed.csv"))
    all_codes = icd_codes + cpt_codes
    chunks = loader.create_chunks_with_metadata(all_codes)

    # Initialize retriever
    retriever = HybridRetriever()
    retriever.initialize_vectorstore(chunks)

    # Test retrieval
    test_queries = [
        "patient with acute chest pain and dyspnea",
        "routine office visit for diabetes management",
        "migraine headache treatment"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query)
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.metadata['code']}: {doc.metadata['description']}")

if __name__ == "__main__":
    main()
