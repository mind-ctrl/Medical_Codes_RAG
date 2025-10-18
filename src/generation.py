"""
Medical response generation with hallucination detection and clinical validation.
"""
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any, Optional
import structlog
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
import re

logger = structlog.get_logger()

@dataclass
class GenerationConfig:
    """Configuration for generation system."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 500
    hallucination_threshold: float = 0.7
    include_rationale: bool = True
    include_citations: bool = True

class MedicalResponseGenerator:
    """Generate medical coding responses with clinical validation."""
    
    def __init__(self, config: GenerationConfig = GenerationConfig()):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Medical coding prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["query", "retrieved_docs", "doc_count"],
            template="""You are an expert medical coding specialist. Based on the provided medical documentation, 
                        identify the most appropriate ICD-10-CM and/or CPT codes for the given clinical scenario.

                        Clinical Query: {query}

                        Retrieved Medical Codes ({doc_count} results):{retrieved_docs}

                        Instructions:
                        1. Analyze the clinical scenario carefully
                        2. Select the most appropriate codes from the retrieved results
                        3. Provide brief medical rationale for each code selection
                        4. Include confidence level (High/Medium/Low) for each recommendation
                        5. Cite specific retrieved sources using [Source X] format
                        6. If no appropriate codes are found, clearly state this

                        Format your response as:
                        **Primary Codes:**
                        - [Code]: [Description] - [Rationale] [Confidence] [Source X]

                        **Additional Considerations:**
                        - Any relevant notes or alternative codes

                        **Confidence Assessment:**
                        Overall confidence in recommendations: [High/Medium/Low]

                        Remember: Only recommend codes that are directly supported by the retrieved documentation."""
        )
    
    def generate_response(self, query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        """Generate medical coding response with validation."""
        logger.info(f"Generating response for query: '{query[:50]}...'")
        
        if not retrieved_docs:
            return {
                "query": query,
                "response": "No relevant medical codes found for the given query.",
                "codes": [],
                "confidence": "Low",
                "hallucination_score": 0.0,
                "sources": []
            }
        
        # Prepare retrieved documents for prompt
        doc_texts = []
        for i, doc in enumerate(retrieved_docs):
            source_info = f"[Source {i+1}] {doc.metadata.get('code', 'N/A')}: {doc.metadata.get('description', doc.page_content)}"
            doc_texts.append(source_info)
        
        retrieved_docs_text = "\n".join(doc_texts)
        
        # Generate response
        try:
            prompt = self.prompt_template.format(
                query=query,
                retrieved_docs=retrieved_docs_text,
                doc_count=len(retrieved_docs)
            )
            
            response = self.llm.invoke(prompt)
            response_text = response.content
            
            # Extract structured information
            extracted_codes = self._extract_codes_from_response(response_text, retrieved_docs)
            confidence = self._extract_confidence(response_text)
            
            # Hallucination detection
            hallucination_score = self._detect_hallucination(query, response_text, retrieved_docs)
            
            # Build final response
            result = {
                "query": query,
                "response": response_text,
                "codes": extracted_codes,
                "confidence": confidence,
                "hallucination_score": hallucination_score,
                "sources": [doc.metadata for doc in retrieved_docs],
                "is_reliable": hallucination_score >= self.config.hallucination_threshold
            }
            
            logger.info(f"Generated response with {len(extracted_codes)} codes, "
                       f"confidence: {confidence}, hallucination_score: {hallucination_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "query": query,
                "response": f"Error generating response: {str(e)}",
                "codes": [],
                "confidence": "Low",
                "hallucination_score": 0.0,
                "sources": [],
                "is_reliable": False
            }
    
    def _extract_codes_from_response(self, response_text: str, retrieved_docs: List[Document]) -> List[Dict[str, Any]]:
        """Extract structured code information from response."""
        codes = []
        
        # Look for code patterns in response
        code_patterns = [
            r'([A-Z]\d{2}\.?\d*)\s*:\s*([^-\n]+)',  # ICD-10 pattern
            r'(\d{5})\s*:\s*([^-\n]+)',             # CPT pattern
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response_text)
            for match in matches:
                code = match[0].strip()
                description = match[1].strip()
                
                # Verify code exists in retrieved documents
                source_doc = None
                for doc in retrieved_docs:
                    if doc.metadata.get('code') == code:
                        source_doc = doc
                        break
                
                if source_doc:
                    codes.append({
                        "code": code,
                        "description": description,
                        "code_type": source_doc.metadata.get('code_type', 'Unknown'),
                        "specialty": source_doc.metadata.get('specialty', 'general'),
                        "verified": True
                    })
        
        return codes
    
    def _extract_confidence(self, response_text: str) -> str:
        """Extract confidence level from response."""
        confidence_patterns = [
            r'Overall confidence[^:]*:\s*(High|Medium|Low)',
            r'Confidence[^:]*:\s*(High|Medium|Low)',
            r'\[?(High|Medium|Low)\]?\s*confidence'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1).title()
        
        return "Medium"  # Default confidence
    
    def _detect_hallucination(self, query: str, response: str, retrieved_docs: List[Document]) -> float:
        """Detect potential hallucinations using semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a medical sentence transformer if available
            model = SentenceTransformer('all-MiniLM-L6-v2')  # Fallback model
            
            # Encode query, response, and retrieved documents
            query_embedding = model.encode([query])
            response_embedding = model.encode([response])
            
            # Encode retrieved documents
            doc_texts = [doc.page_content for doc in retrieved_docs]
            if doc_texts:
                doc_embeddings = model.encode(doc_texts)
                
                # Calculate similarity between response and retrieved docs
                doc_similarities = cosine_similarity(response_embedding, doc_embeddings)[0]
                max_doc_similarity = np.max(doc_similarities)
                
                # Calculate similarity between response and query
                query_similarity = cosine_similarity(response_embedding, query_embedding)[0][0]
                
                # Combined hallucination score (higher = less hallucination)
                hallucination_score = 0.7 * max_doc_similarity + 0.3 * query_similarity
                
                return float(hallucination_score)
            else:
                return 0.5  # Neutral score when no docs available
                
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return 0.5  # Default neutral score
    
    def generate_batch_responses(self, queries_and_docs: List[tuple[str, List[Document]]]) -> List[Dict[str, Any]]:
        """Generate responses for multiple queries."""
        results = []
        
        for query, docs in queries_and_docs:
            result = self.generate_response(query, docs)
            results.append(result)
        
        logger.info(f"Generated {len(results)} batch responses")
        return results

def main():
    """Demo generation system."""
    # Sample retrieved documents
    sample_docs = [
        Document(
            page_content="R07.9: Chest pain, unspecified",
            metadata={
                "code": "R07.9",
                "description": "Chest pain, unspecified",
                "code_type": "ICD10",
                "specialty": "cardiology"
            }
        ),
        Document(
            page_content="I20.0: Unstable angina",
            metadata={
                "code": "I20.0", 
                "description": "Unstable angina",
                "code_type": "ICD10",
                "specialty": "cardiology"
            }
        ),
        Document(
            page_content="99214: Office visit for established patient",
            metadata={
                "code": "99214",
                "description": "Office or other outpatient visit for evaluation and management of established patient",
                "code_type": "CPT",
                "specialty": "general"
            }
        )
    ]
    
    generator = MedicalResponseGenerator()
    
    test_query = "patient with acute chest pain and dyspnea"
    result = generator.generate_response(test_query, sample_docs)
    
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
    print(f"Extracted Codes: {result['codes']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Hallucination Score: {result['hallucination_score']:.3f}")
    print(f"Is Reliable: {result['is_reliable']}")

if __name__ == "__main__":
    main()
