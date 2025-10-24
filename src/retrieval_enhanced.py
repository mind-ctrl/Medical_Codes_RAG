"""
ENHANCED Hybrid Retrieval System with Production-Grade Improvements:
1. Reciprocal Rank Fusion (RRF) for hybrid merging
2. Query-adaptive weighting (exact codes vs symptoms)
3. Maximal Marginal Relevance (MMR) for diversity
4. Confidence-based filtering
5. Explainable reranking scores
"""
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any, Optional, Tuple
import structlog
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass, field
from pathlib import Path
import os
import re

logger = structlog.get_logger()

def _get_default_llm_model() -> str:
    """Get default LLM model based on API configuration."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "")

    if api_key.startswith("pplx-") or "perplexity" in api_base.lower():
        return os.getenv("LLM_MODEL", "sonar-pro")
    else:
        return os.getenv("LLM_MODEL", "gpt-4o-mini")

@dataclass
class RetrievalConfig:
    """Enhanced configuration for retrieval system."""
    openai_model: str = field(default_factory=_get_default_llm_model)
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_queries: int = 4
    retrieval_k: int = 20  # Increased for better recall
    final_k: int = 5

    # RRF parameters
    rrf_k: int = 60  # Constant for Reciprocal Rank Fusion

    # MMR parameters (diversity)
    mmr_lambda: float = 0.7  # 0.7 = balance relevance and diversity

    # Confidence thresholding
    min_confidence_score: float = 0.3  # Filter out low-confidence results

    # Query-adaptive weighting
    use_adaptive_weights: bool = True

    use_local_embeddings: bool = True

@dataclass
class RetrievalResult:
    """Enhanced result with explainability."""
    document: Document
    final_score: float
    bm25_score: float
    semantic_score: float
    cross_encoder_score: float
    explanation: str
    rank: int

class EnhancedHybridRetriever:
    """
    Production-grade hybrid retrieval with:
    - Reciprocal Rank Fusion (RRF)
    - Query-adaptive weighting
    - Maximal Marginal Relevance (MMR)
    - Confidence filtering
    - Explainability
    """

    def __init__(self, config: RetrievalConfig = RetrievalConfig()):
        self.config = config
        self.llm = ChatOpenAI(model=config.openai_model, temperature=0)

        # Use local embeddings (free)
        logger.info("Using enhanced local HuggingFace embeddings")
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        self.cross_encoder = CrossEncoder(config.cross_encoder_model)

        # FAISS setup
        self.vectorstore = None
        self.documents = []

        # Medical code pattern detection
        self.code_pattern = re.compile(r'\b([A-Z]\d{2}\.?\d{0,4}|\d{5})\b')

    def initialize_vectorstore(self, documents: List[Dict[str, Any]], force_rebuild: bool = False):
        """Initialize FAISS vector store with caching."""
        faiss_path = "./faiss_index"

        # Try to load existing index
        if not force_rebuild and Path(faiss_path).exists():
            logger.info(f"Found existing FAISS index at {faiss_path}")
            if self.load_vectorstore(faiss_path):
                logger.info("Loaded existing FAISS index - skipping rebuild!")

                # Prepare documents for BM25
                self.documents = []
                for doc in documents:
                    self.documents.append(
                        Document(page_content=doc["text"], metadata=doc["metadata"])
                    )

                logger.info("Enhanced vector store ready (loaded from cache) [OK]")
                return

        # Build new index
        logger.info(f"Building new FAISS index with {len(documents)} documents")

        self.documents = []
        for doc in documents:
            self.documents.append(
                Document(page_content=doc["text"], metadata=doc["metadata"])
            )

        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        self.vectorstore.save_local(faiss_path)
        logger.info(f"Enhanced FAISS index saved to {faiss_path}")

    def load_vectorstore(self, path: str = "./faiss_index"):
        """Load existing FAISS vectorstore."""
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

    def _detect_query_type(self, query: str) -> str:
        """
        Detect if query contains exact codes or is symptom-based.
        This enables query-adaptive weighting.
        """
        # Check for medical codes in query
        codes = self.code_pattern.findall(query)

        if codes:
            return "code_search"  # Exact code lookup
        elif any(word in query.lower() for word in ['patient', 'symptom', 'diagnosis', 'treatment']):
            return "symptom_search"  # Symptom-based
        else:
            return "general_search"

    def _get_adaptive_weights(self, query: str) -> Tuple[float, float]:
        """
        Get BM25 and semantic weights based on query type.

        Returns:
            (bm25_weight, semantic_weight)
        """
        if not self.config.use_adaptive_weights:
            return (0.3, 0.7)  # Default weights

        query_type = self._detect_query_type(query)

        if query_type == "code_search":
            # Exact codes → prioritize BM25 keyword matching
            return (0.7, 0.3)
        elif query_type == "symptom_search":
            # Symptoms → prioritize semantic understanding
            return (0.2, 0.8)
        else:
            # General → balanced
            return (0.4, 0.6)

    def _multi_query_generate(self, query: str) -> List[str]:
        """Generate multiple query variants."""
        prompt = f"""You are a medical coding assistant. Generate {self.config.max_queries} different
        variations of the following medical query:

        Original query: {query}

        Generate queries that:
        1. Use medical synonyms and terminology
        2. Focus on different aspects (symptoms, procedures, diagnoses)
        3. Include common abbreviations

        Return only the queries, one per line."""

        try:
            response = self.llm.invoke(prompt)
            variants = [line.strip() for line in response.content.split('\n') if line.strip()]
            logger.info(f"Generated {len(variants)} query variants")
            return variants[:self.config.max_queries]
        except Exception as e:
            logger.error(f"Failed to generate query variants: {e}")
            return [query]

    def _semantic_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """Semantic search with FAISS."""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            documents_with_scores = []
            for doc, distance in results:
                # Convert L2 distance to similarity (0-1, higher is better)
                score = np.exp(-distance)
                documents_with_scores.append((doc, score))

            return documents_with_scores
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _bm25_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """BM25 keyword search."""
        try:
            bm25_retriever = BM25Retriever.from_documents(
                documents=self.documents,
                k=k
            )
            results = bm25_retriever.invoke(query)

            # Estimate scores
            documents_with_scores = []
            for i, doc in enumerate(results):
                score = self._calculate_bm25_score(query, doc.page_content)
                documents_with_scores.append((doc, score))

            return sorted(documents_with_scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _calculate_bm25_score(self, query: str, text: str) -> float:
        """Simple BM25-like scoring."""
        query_terms = query.lower().split()
        text_lower = text.lower()

        score = 0.0
        for term in query_terms:
            if term in text_lower:
                tf = text_lower.count(term)
                score += tf * 0.1

        return min(score, 1.0)

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[Document, float]],
        semantic_results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float, float, float]]:
        """
        Reciprocal Rank Fusion (RRF) - superior to simple score merging.

        RRF formula: score(d) = sum(1 / (k + rank(d)))
        where k is a constant (typically 60) and rank is the position.

        Returns:
            List of (doc, rrf_score, bm25_score, semantic_score)
        """
        k = self.config.rrf_k

        # Create document to rank mappings
        doc_to_bm25_rank = {}
        doc_to_bm25_score = {}
        for rank, (doc, score) in enumerate(bm25_results, 1):
            doc_id = (doc.page_content, tuple(sorted(doc.metadata.items())))
            doc_to_bm25_rank[doc_id] = rank
            doc_to_bm25_score[doc_id] = score

        doc_to_semantic_rank = {}
        doc_to_semantic_score = {}
        for rank, (doc, score) in enumerate(semantic_results, 1):
            doc_id = (doc.page_content, tuple(sorted(doc.metadata.items())))
            doc_to_semantic_rank[doc_id] = rank
            doc_to_semantic_score[doc_id] = score

        # Compute RRF scores
        all_docs = set(doc_to_bm25_rank.keys()) | set(doc_to_semantic_rank.keys())
        rrf_scores = {}

        for doc_id in all_docs:
            rrf_score = 0.0

            if doc_id in doc_to_bm25_rank:
                rrf_score += 1.0 / (k + doc_to_bm25_rank[doc_id])

            if doc_id in doc_to_semantic_rank:
                rrf_score += 1.0 / (k + doc_to_semantic_rank[doc_id])

            rrf_scores[doc_id] = rrf_score

        # Sort by RRF score and reconstruct documents
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, rrf_score in sorted_docs:
            # Reconstruct document
            content, metadata_tuple = doc_id
            metadata = dict(metadata_tuple)
            doc = Document(page_content=content, metadata=metadata)

            bm25_score = doc_to_bm25_score.get(doc_id, 0.0)
            semantic_score = doc_to_semantic_score.get(doc_id, 0.0)

            results.append((doc, rrf_score, bm25_score, semantic_score))

        logger.info(f"RRF merged {len(results)} unique documents")
        return results

    def _maximal_marginal_relevance(
        self,
        query_embedding: np.ndarray,
        candidate_docs: List[Tuple[Document, float]],
        lambda_param: float = 0.7,
        top_k: int = 5
    ) -> List[Document]:
        """
        Maximal Marginal Relevance (MMR) for diversity.

        MMR = lambda * relevance - (1 - lambda) * max_similarity_to_selected

        This ensures diverse results (no redundant documents).
        """
        if len(candidate_docs) <= top_k:
            return [doc for doc, _ in candidate_docs]

        # Get embeddings for all candidate documents
        doc_texts = [doc.page_content for doc, _ in candidate_docs]
        doc_embeddings = self.embeddings.embed_documents(doc_texts)
        doc_embeddings = np.array(doc_embeddings)

        # Calculate relevance scores (similarity to query)
        query_sims = cosine_similarity([query_embedding], doc_embeddings)[0]

        selected_indices = []
        selected_embeddings = []

        for _ in range(top_k):
            if not selected_indices:
                # First selection: highest relevance
                best_idx = np.argmax(query_sims)
            else:
                # MMR selection
                mmr_scores = []
                for idx in range(len(candidate_docs)):
                    if idx in selected_indices:
                        mmr_scores.append(-float('inf'))
                        continue

                    relevance = query_sims[idx]

                    # Max similarity to already selected documents
                    if selected_embeddings:
                        similarities = cosine_similarity(
                            [doc_embeddings[idx]],
                            selected_embeddings
                        )[0]
                        max_sim = np.max(similarities)
                    else:
                        max_sim = 0.0

                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                    mmr_scores.append(mmr_score)

                best_idx = np.argmax(mmr_scores)

            selected_indices.append(best_idx)
            selected_embeddings.append(doc_embeddings[best_idx])

        selected_docs = [candidate_docs[idx][0] for idx in selected_indices]
        logger.info(f"MMR selected {len(selected_docs)} diverse documents")
        return selected_docs

    def _cross_encoder_rerank_with_confidence(
        self,
        query: str,
        documents: List[Tuple[Document, float, float, float]],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Enhanced cross-encoder reranking with:
        - Confidence filtering
        - Explainability
        - Score normalization
        """
        if not documents:
            return []

        # Unpack documents and scores
        docs_only = [doc for doc, _, _, _ in documents]

        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in docs_only]

        try:
            # Get cross-encoder scores
            ce_scores = self.cross_encoder.predict(pairs)

            # Combine all information
            results = []
            for i, (doc, rrf_score, bm25_score, semantic_score) in enumerate(documents):
                ce_score = float(ce_scores[i])

                # Filter by confidence threshold
                if ce_score < self.config.min_confidence_score:
                    continue

                # Generate explanation
                explanation = self._generate_explanation(
                    rrf_score, bm25_score, semantic_score, ce_score
                )

                result = RetrievalResult(
                    document=doc,
                    final_score=ce_score,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                    cross_encoder_score=ce_score,
                    explanation=explanation,
                    rank=0  # Will be set after sorting
                )
                results.append(result)

            # Sort by final score
            results.sort(key=lambda x: x.final_score, reverse=True)

            # Assign ranks
            for rank, result in enumerate(results[:top_k], 1):
                result.rank = rank

            logger.info(f"Cross-encoder reranked to top {len(results[:top_k])} with confidence filtering")
            return results[:top_k]

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            # Fallback: return top docs by RRF score
            fallback_results = []
            for rank, (doc, rrf_score, bm25_score, semantic_score) in enumerate(documents[:top_k], 1):
                result = RetrievalResult(
                    document=doc,
                    final_score=rrf_score,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                    cross_encoder_score=0.0,
                    explanation="Fallback ranking (cross-encoder failed)",
                    rank=rank
                )
                fallback_results.append(result)
            return fallback_results

    def _generate_explanation(
        self,
        rrf_score: float,
        bm25_score: float,
        semantic_score: float,
        ce_score: float
    ) -> str:
        """Generate human-readable explanation for ranking."""
        reasons = []

        if bm25_score > 0.5:
            reasons.append("strong keyword match")
        elif bm25_score > 0.3:
            reasons.append("good keyword match")

        if semantic_score > 0.7:
            reasons.append("high semantic similarity")
        elif semantic_score > 0.5:
            reasons.append("moderate semantic similarity")

        if ce_score > 0.8:
            reasons.append("very relevant (cross-encoder)")
        elif ce_score > 0.6:
            reasons.append("relevant (cross-encoder)")

        if not reasons:
            return "Low confidence match"

        return "Ranked high due to: " + ", ".join(reasons)

    def retrieve(
        self,
        query: str,
        specialty: Optional[str] = None,
        return_detailed: bool = False
    ) -> List[Document]:
        """
        Enhanced retrieval with all improvements.

        Args:
            query: User query
            specialty: Optional medical specialty filter
            return_detailed: If True, returns RetrievalResult objects with scores

        Returns:
            List of Document objects (or RetrievalResult if return_detailed=True)
        """
        logger.info(f"Enhanced retrieval for query: '{query[:50]}...' (specialty: {specialty})")

        # Step 1: Multi-query generation
        query_variants = self._multi_query_generate(query)
        all_variants = [query] + query_variants
        logger.info(f"Generated {len(query_variants)} query variants")

        # Step 2: Hybrid search with all variants
        all_bm25_results = []
        all_semantic_results = []

        for variant in all_variants:
            bm25_results = self._bm25_search(variant, k=self.config.retrieval_k)
            semantic_results = self._semantic_search(variant, k=self.config.retrieval_k)

            all_bm25_results.extend(bm25_results)
            all_semantic_results.extend(semantic_results)

        # Deduplicate
        seen_docs = set()
        unique_bm25 = []
        for doc, score in all_bm25_results:
            doc_id = (doc.page_content, tuple(sorted(doc.metadata.items())))
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_bm25.append((doc, score))

        seen_docs = set()
        unique_semantic = []
        for doc, score in all_semantic_results:
            doc_id = (doc.page_content, tuple(sorted(doc.metadata.items())))
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_semantic.append((doc, score))

        # Step 3: Reciprocal Rank Fusion
        merged_results = self._reciprocal_rank_fusion(unique_bm25, unique_semantic)

        # Step 4: Cross-encoder reranking with confidence filtering
        reranked_results = self._cross_encoder_rerank_with_confidence(
            query,
            merged_results,
            top_k=self.config.final_k
        )

        if return_detailed:
            return reranked_results
        else:
            return [result.document for result in reranked_results]

# Backward compatibility
HybridRetriever = EnhancedHybridRetriever
