"""
Medical data loader with UMLS integration and specialty metadata.
Handles ICD-10-CM and CPT code processing with semantic enrichment.
"""
import pandas as pd
import requests
import zipfile
import spacy
from pathlib import Path
from typing import List, Dict, Any, Optional
import structlog
from pydantic import BaseModel
import re

logger = structlog.get_logger()

class MedicalCode(BaseModel):
    """Medical code data structure with metadata."""
    code: str
    description: str
    code_type: str  # 'ICD10' or 'CPT'
    specialty: Optional[str] = None
    umls_concepts: List[str] = []
    synonyms: List[str] = []

class MedicalDataLoader:
    """Enhanced medical data loader (UMLS integration disabled by default)."""

    def __init__(self, data_dir: Path = Path("data"), enable_umls: bool = False):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        self.nlp = None
        self.enable_umls = enable_umls
        self._umls_linker_missing_reported = False

        # Only load scispacy if explicitly enabled
        if self.enable_umls:
            self._load_scispacy_model()

    def _load_scispacy_model(self):
        """Load scispacy model for UMLS entity linking (optional feature)."""
        try:
            # Download model if not exists
            import subprocess
            import sys

            try:
                self.nlp = spacy.load("en_core_sci_sm")
            except OSError:
                logger.info("Downloading scispacy model...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install",
                    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
                ])
                self.nlp = spacy.load("en_core_sci_sm")

            # Add entity linker for UMLS
            self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
            logger.info("ScispaCy model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load scispacy model: {e}")
            logger.info("UMLS enrichment will be skipped")
            self.nlp = None
    
    def download_icd10_data(self) -> Path:
        """Use local ICD-10-CM codes file."""
        txt_path = self.data_dir / "icd10cm-codes-2025.txt"  # No .txt extension based on your screenshot
        
        if not txt_path.exists():
            raise FileNotFoundError(
                f"ICD-10-CM codes file not found at {txt_path}. "
                "Please download from CDC and place in data/ folder."
            )
        
        logger.info(f"Using local ICD-10-CM file: {txt_path}")
        return txt_path
    
    def download_cpt_data(self) -> Path:
        """Use local CPT codes Excel file."""
        excel_path = self.data_dir / "opc-cpt-pcm-nhsn.xlsx"  # Based on your screenshot
            
        if not excel_path.exists():
            raise FileNotFoundError(
                f"CPT codes file not found at {excel_path}. "
                "Please download from CDC NHSN and place in data/ folder."
            )
            
        logger.info(f"Using local CPT file: {excel_path}")
        return excel_path

    
    def parse_icd10_codes(self, file_path: Path) -> List[MedicalCode]:
        """Parse ICD-10-CM codes from text file."""
        logger.info("Parsing ICD-10-CM codes from local file...")
        codes = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty/comment lines
                    continue
                
                # Format: "CODE  Description"
                # Split on first whitespace, allowing multi-word descriptions
                parts = line.split(maxsplit=1)
                if len(parts) >= 2:
                    code = parts[0]
                    description = parts[1]
                    
                    # Enhance with UMLS (if enabled) or skip
                    enhanced = self._enhance_with_umls(
                        code, description, "ICD10", specialty=None
                    )
                    codes.append(enhanced)
        
        logger.info(f"Parsed {len(codes)} ICD-10-CM codes")
        return codes
    
    def parse_cpt_codes(self, file_path: Path) -> List[MedicalCode]:
        """Parse CPT codes from Excel file."""
        logger.info("Parsing CPT codes from local Excel file...")
        codes = []
        
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            
            # Inspect columns - common names are 'CPT Code', 'Code', 'Description', 'Procedure'
            # Adjust column names based on actual structure
            code_col = None
            desc_col = None
            
            for col in df.columns:
                if 'code' in col.lower() or 'cpt' in col.lower():
                    code_col = col
                if 'desc' in col.lower() or 'procedure' in col.lower():
                    desc_col = col
            
            if not code_col or not desc_col:
                logger.warning(f"Could not find code/description columns. Available: {list(df.columns)}")
                return codes
            
            for _, row in df.iterrows():
                code = str(row[code_col]).strip()
                description = str(row[desc_col]).strip()
                
                if code and code != 'nan':
                    enhanced = self._enhance_with_umls(
                        code, description, "CPT", specialty=None
                    )
                    codes.append(enhanced)
            
            logger.info(f"Parsed {len(codes)} CPT codes")
            
        except Exception as e:
            logger.error(f"Error parsing CPT Excel: {e}")
        
        return codes
    
    def _enhance_with_umls(
        self, code: str, description: str, code_type: str, specialty: str = None
    ) -> MedicalCode:
        """
        Enhance medical codes with UMLS concepts and synonyms (only if enabled).
        UMLS is disabled by default - set enable_umls=True in __init__ to enable.
        """
        umls_concepts: List[str] = []
        synonyms: List[str] = []

        # Skip UMLS enrichment if not enabled or NLP model not loaded
        if self.enable_umls and self.nlp:
            try:
                doc = self.nlp(description)

                # Extract up to 3 UMLS concepts per entity
                for ent in doc.ents:
                    for concept_id, _ in getattr(ent._, "kb_ents", [])[:3]:
                        umls_concepts.append(concept_id)

                # Attempt to add synonyms via scispacy_linker
                try:
                    linker = self.nlp.get_pipe("scispacy_linker")
                    for concept_id in umls_concepts[:2]:
                        concept = linker.kb.cui_to_entity.get(concept_id)
                        if concept and hasattr(concept, "aliases"):
                            synonyms.extend(concept.aliases[:5])
                except KeyError:
                    if not self._umls_linker_missing_reported:
                        logger.warning("scispacy_linker not found; skipping UMLS enrichment")
                        self._umls_linker_missing_reported = True
                except Exception as e:
                    logger.error(f"Error during UMLS enrichment: {e}")
            except Exception as e:
                logger.error(f"UMLS processing error: {e}")

        return MedicalCode(
            code=code,
            description=description,
            code_type=code_type,
            specialty=specialty,
            umls_concepts=umls_concepts,
            synonyms=list(set(synonyms)),
        )
    
    def create_chunks_with_metadata(self, codes: List[MedicalCode]) -> List[Dict[str, Any]]:
        """Create document chunks with rich metadata for retrieval."""
        chunks = []
        
        for code in codes:
            # Primary chunk with full description
            chunk_text = f"{code.code}: {code.description}"
            
            # Add synonyms to chunk for better retrieval
            if code.synonyms:
                chunk_text += f" | Synonyms: {', '.join(code.synonyms[:3])}"
            
            chunk = {
                "text": chunk_text,
                "metadata": {
                    "code": code.code,
                    "code_type": code.code_type,
                    "specialty": code.specialty or "general",
                    "umls_concepts": code.umls_concepts,
                    "synonyms": code.synonyms,
                    "description": code.description
                }
            }
            chunks.append(chunk)
            
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks
    
    def save_processed_data(self, codes: List[MedicalCode], filename: str):
        """Save processed codes to CSV for analysis."""
        df_data = []
        for code in codes:
            df_data.append({
            "code": code.code,
            "description": code.description,
            "code_type": code.code_type,
            "specialty": code.specialty,
            "umls_concepts": "|".join(code.umls_concepts),
            "synonyms": "|".join(code.synonyms)
        })
    
            df = pd.DataFrame(df_data)
            output_path = self.data_dir / filename
        
            # DEBUG: Print what we're about to save
            print(f"DEBUG: DataFrame has {len(df)} rows")
            print(f"DEBUG: Saving to: {output_path.resolve()}")
        try:
            df.to_csv(output_path, index=False)
            print(f"DEBUG: File saved successfully!")
            logger.info(f"Saved processed data: {output_path}")
        except Exception as e:
            print(f"ERROR saving file: {e}")
            raise
        
        return output_path


def main():
        """Demo data loading pipeline."""
        loader = MedicalDataLoader()
        
        # Download and process ICD-10 codes
        icd_file = loader.download_icd10_data()
        icd_codes = loader.parse_icd10_codes(icd_file)
        loader.save_processed_data(icd_codes, "icd10_processed.csv")
        
        # Download and process CPT codes  
        cpt_file = loader.download_cpt_data()
        cpt_codes = loader.parse_cpt_codes(cpt_file)
        loader.save_processed_data(cpt_codes, "cpt_processed.csv")
        
        # Create chunks for retrieval
        all_codes = icd_codes + cpt_codes
        chunks = loader.create_chunks_with_metadata(all_codes)
        
        print(f"Successfully processed {len(all_codes)} medical codes")
        print(f"Sample chunk: {chunks[0]}")

if __name__ == "__main__":
        main()
