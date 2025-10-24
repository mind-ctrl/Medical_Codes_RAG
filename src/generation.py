"""
Medical response generation with hallucination detection and clinical validation.
"""
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any, Optional
import json
import structlog
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass, field
import re
import os

logger = structlog.get_logger()

def _get_default_model() -> str:
    """Get default model based on API configuration."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "")

    # Check if using Perplexity
    if api_key.startswith("pplx-") or "perplexity" in api_base.lower():
        return os.getenv("LLM_MODEL", "sonar-pro")
    else:
        return os.getenv("LLM_MODEL", "gpt-4o-mini")

@dataclass
class GenerationConfig:
    """Configuration for generation system."""
    model: str = field(default_factory=_get_default_model)
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
        """Extract structured code information from response with improved parsing."""
        codes = []

        # Enhanced code patterns with multiple formats
        icd10_patterns = [
            r'ICD-?10[:\-\s]*([A-Z]\d{2}\.?\d{0,4})',  # ICD-10: A12.34
            r'\b([A-Z]\d{2}\.?\d{1,4})\b',              # Direct code A12.34
            r'([A-Z]\d{2})\b',                          # Short code A12
            r'code[:\s]+([A-Z]\d{2}\.?\d{0,4})',       # "code A12.34"
        ]

        cpt_patterns = [
            r'CPT[:\-\s]*(\d{5})',                     # CPT: 12345
            r'\b(\d{5})\b',                             # Direct code 12345
            r'code[:\s]+(\d{5})',                       # "code 12345"
        ]

        # Extract ICD-10 codes
        for pattern in icd10_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                code = match.upper().strip() if isinstance(match, str) else match[0].upper().strip()
                # Normalize format (ensure dot if more than 3 chars)
                if len(code.replace('.', '')) > 3 and '.' not in code:
                    code = code[:3] + '.' + code[3:]

                # Verify and add
                source_doc = self._find_code_in_docs(code, retrieved_docs, 'ICD10')
                if source_doc:
                    codes.append({
                        "code": code,
                        "description": source_doc.metadata.get('description', 'N/A'),
                        "code_type": "ICD10",
                        "specialty": source_doc.metadata.get('specialty', 'general'),
                        "verified": True
                    })

        # Extract CPT codes
        for pattern in cpt_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                code = match.strip() if isinstance(match, str) else match[0].strip()

                # Verify and add
                source_doc = self._find_code_in_docs(code, retrieved_docs, 'CPT')
                if source_doc:
                    codes.append({
                        "code": code,
                        "description": source_doc.metadata.get('description', 'N/A'),
                        "code_type": "CPT",
                        "specialty": source_doc.metadata.get('specialty', 'general'),
                        "verified": True
                    })

        # Deduplicate codes
        unique_codes = []
        seen_codes = set()
        for code_obj in codes:
            code_key = code_obj['code']
            if code_key not in seen_codes:
                seen_codes.add(code_key)
                unique_codes.append(code_obj)

        # If no codes extracted, try LLM-based structured extraction
        if not unique_codes:
            unique_codes = self._llm_extract_codes(response_text, retrieved_docs)

        return unique_codes

    def _find_code_in_docs(self, code: str, docs: List[Document], code_type: str = None) -> Optional[Document]:
        """Find a code in retrieved documents with fuzzy matching."""
        # Exact match
        for doc in docs:
            doc_code = doc.metadata.get('code', '')
            if doc_code == code:
                return doc

        # Fuzzy match (without dots/spaces)
        normalized_code = code.replace('.', '').replace(' ', '').upper()
        for doc in docs:
            doc_code = doc.metadata.get('code', '').replace('.', '').replace(' ', '').upper()
            if doc_code == normalized_code:
                if code_type is None or doc.metadata.get('code_type') == code_type:
                    return doc

        return None

    def _llm_extract_codes(self, response_text: str, retrieved_docs: List[Document]) -> List[Dict[str, Any]]:
        """Use LLM to extract codes if regex fails."""
        try:
            # Build a structured extraction prompt
            doc_codes = [doc.metadata.get('code', 'N/A') for doc in retrieved_docs[:10]]

            extraction_prompt = f"""Extract ONLY the medical codes mentioned in this response.
Response: {response_text[:500]}

Available codes from retrieval: {', '.join(doc_codes)}

Return codes in JSON format:
{{"codes": ["CODE1", "CODE2"]}}

Extract codes:"""

            extraction_response = self.llm.invoke(extraction_prompt)
            extraction_text = extraction_response.content

            # Parse JSON response
            import json
            try:
                # Try to find JSON in response
                json_match = re.search(r'\{.*\}', extraction_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    extracted = data.get('codes', [])

                    # Verify each code
                    verified_codes = []
                    for code in extracted:
                        source_doc = self._find_code_in_docs(code, retrieved_docs)
                        if source_doc:
                            verified_codes.append({
                                "code": code,
                                "description": source_doc.metadata.get('description', 'N/A'),
                                "code_type": source_doc.metadata.get('code_type', 'Unknown'),
                                "specialty": source_doc.metadata.get('specialty', 'general'),
                                "verified": True
                            })

                    return verified_codes
            except json.JSONDecodeError:
                logger.warning("LLM extraction returned invalid JSON")

        except Exception as e:
            logger.error(f"LLM code extraction failed: {e}")

        return []
    
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
