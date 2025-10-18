"""
Advanced hybrid retrieval system with UMLS expansion, 
multi-query generation, and specialty clustering.
"""
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any, Optional, Tuple
import structlog
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.config import Settings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from dataclasses import dataclass
import asyncio
from pathlib import Path

logger = structlog.get_logger()

@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_queries: int = 3
    retrieval_k: int = 10
    final_k: int = 5
    cluster_k: int = 5
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7

class UMLSQueryExpander:
    """Expand queries using UMLS medical synonyms."""
    
    def __init__(self):
        self.nlp = None
        self._load_scispacy()
    
    def _load_scispacy(self):
        """Load scispacy with UMLS linker."""
        try:
            self.nlp = spacy.load("en_core_sci_sm")
            if "scispacy_linker" not in self.nlp.pipe_names:
                self.nlp.add_pipe("scispacy_linker", config={
                    "resolve_abbreviations": True, 
                    "linker_name": "umls"
                })
        except Exception as e:
            logger.error(f"Failed to load scispacy: {e}")
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with medical synonyms."""
        if not self.nlp:
            return [query]
        
        expansions = [query]
        doc = self.nlp(query)
        
        # Extract medical entities and their synonyms
        linker = self.nlp.get_pipe("scispacy_linker")
        
        for ent in doc.ents:
            if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                for kb_ent in ent._.kb_ents[:2]:  # Top 2 concepts
                    try:
                        concept = linker.kb.cui_to_entity.get(kb_ent[0])
                        if concept and hasattr(concept, 'aliases'):
                            # Add relevant synonyms
                            for alias in concept.aliases[:3]:
                                if alias.lower() != ent.text.lower():
                                    expanded_query = query.replace(ent.text, alias)
                                    expansions.append(expanded_query)
                    except:
                        continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_expansions = []
        for exp in expansions:
            if exp not in seen:
                seen.add(exp)
                unique_expansions.append(exp)
        
        logger.info(f"Expanded query '{query}' to {len(unique_expansions)} variants")
        return unique_expansions[:5]  # Limit expansions

class HybridRetriever:
    """Advanced hybrid retrieval with multiple enhancement stages."""
    
    def __init__(self, config: RetrievalConfig = RetrievalConfig()):
        self.config = config
        self.llm = ChatOpenAI(model=config.openai_model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=config.embedding_model)
        self.cross_encoder = CrossEncoder(config.cross_encoder_model)
        self.umls_expander = UMLSQueryExpander()
        
        # ChromaDB setup
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_data",
            settings=Settings(allow_reset=True)
        )
        self.collection = None
        self.documents = []
        self.clusterer = None
        
    def initialize_vectorstore(self, documents: List[Dict[str, Any]]):
        """Initialize ChromaDB with documents."""
        logger.info(f"Initializing vector store with {len(documents)} documents")

        # Create or recreate the collection
        collection_name = "medical_codes"
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        texts = []
        raw_metadatas = []
        ids = []
        self.documents = []

        for i, doc in enumerate(documents):
            texts.append(doc["text"])
            raw_metadatas.append(doc["metadata"])
            ids.append(f"doc_{i}")
            self.documents.append(Document(page_content=doc["text"], metadata=doc["metadata"]))

        # Clean metadata: flatten lists into strings
        cleaned_metadatas = []
        for meta in raw_metadatas:
            cm = {}
            for key, value in meta.items():
                if isinstance(value, list):
                    # join list elements into a single string
                    cm[key] = "|".join(str(v) for v in value)
                else:
                    cm[key] = value
            cleaned_metadatas.append(cm)

        # Generate embeddings
        embeddings_list = self.embeddings.embed_documents(texts)

        # Add to ChromaDB in manageable batches
        MAX_BATCH = 5000  # must be <= 5461
        for start in range(0, len(texts), MAX_BATCH):
                end = start + MAX_BATCH
                self.collection.add(
                    documents=texts[start:end],
                    metadatas=cleaned_metadatas[start:end],
                    ids=ids[start:end],
                    embeddings=embeddings_list[start:end],
                )
                print(f"Added batch {start // MAX_BATCH + 1} ({end - start} items)")  # Optional: for progress tracking
                
        # Optionally cluster for specialty filtering
        self._initialize_clustering(embeddings_list, cleaned_metadatas)

        logger.info("Vector store initialized successfully")
    
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
        
        logger.info(f"Initialized clustering with {self.config.cluster_k} specialty clusters")
    
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
        """Perform semantic search using ChromaDB."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            documents_with_scores = []
            for i, (doc_text, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                score = 1.0 - distance  # Convert distance to similarity
                doc = Document(page_content=doc_text, metadata=metadata)
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
            
            logger.info(f"Cross-encoder reranked {len(documents)} to top {len(reranked_docs)}")
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
        
        # Stage 1: Query expansion with UMLS
        expanded_queries = self.umls_expander.expand_query(query)
        
        # Stage 2: Multi-query generation
        llm_queries = self._multi_query_generate(query)
        
        # Combine all query variants
        all_queries = list(set(expanded_queries + llm_queries))
        logger.info(f"Generated {len(all_queries)} query variants")
        
        # Stage 3: Hybrid retrieval for each query
        all_semantic_results = []
        all_bm25_results = []
        
        for q in all_queries:
            semantic_results = self._semantic_search(q, k=self.config.retrieval_k)
            bm25_results = self._bm25_search(q, k=self.config.retrieval_k)
            
            all_semantic_results.extend(semantic_results)
            all_bm25_results.extend(bm25_results)
        
        # Stage 4: Fusion and deduplication
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
        
        # Stage 5: Specialty filtering
        if specialty_filter:
            candidate_docs = self._filter_by_specialty(candidate_docs, specialty_filter)
        
        # Stage 6: Cross-encoder reranking
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
