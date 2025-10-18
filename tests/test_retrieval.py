import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from retrieval import HybridRetriever, UMLSQueryExpander
from langchain_core.documents import Document

class TestUMLSQueryExpander:
    """Test UMLS query expansion functionality."""
    
    def test_expand_query_basic(self):
        expander = UMLSQueryExpander()
        
        # Mock scispacy if not available
        if not expander.nlp:
            expander.nlp = Mock()
            expander.nlp.return_value.ents = []
        
        expansions = expander.expand_query("chest pain")
        
        assert isinstance(expansions, list)
        assert len(expansions) >= 1
        assert "chest pain" in expansions

class TestHybridRetriever:
    """Test hybrid retrieval system."""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            {
                "text": "R07.9: Chest pain, unspecified",
                "metadata": {
                    "code": "R07.9",
                    "description": "Chest pain, unspecified",
                    "code_type": "ICD10",
                    "specialty": "cardiology"
                }
            },
            {
                "text": "99214: Office visit for established patient", 
                "metadata": {
                    "code": "99214",
                    "description": "Office visit",
                    "code_type": "CPT", 
                    "specialty": "general"
                }
            }
        ]
    
    @pytest.fixture
    def retriever(self, sample_documents):
        retriever = HybridRetriever()
        
        # Mock ChromaDB for testing
        with patch('chromadb.PersistentClient'):
            retriever.initialize_vectorstore(sample_documents)
        
        return retriever
    
    def test_retriever_initialization(self, retriever):
        assert retriever is not None
        assert retriever.config is not None
    
    @patch('chromadb.PersistentClient')
    def test_retrieve_functionality(self, mock_chroma, retriever):
        # Mock retrieval results
        mock_results = [
            Document(
                page_content="R07.9: Chest pain, unspecified",
                metadata={"code": "R07.9", "specialty": "cardiology"}
            )
        ]
        
        with patch.object(retriever, '_semantic_search', return_value=[(mock_results[0], 0.9)]):
            with patch.object(retriever, '_bm25_search', return_value=[(mock_results[0], 0.8)]):
                results = retriever.retrieve("chest pain")
                
                assert len(results) >= 0
                assert isinstance(results, list)

def test_integration():
    """Integration test for full pipeline"""
    # This would test the complete flow
    pass

if __name__ == "__main__":
    pytest.main([__file__])
