"""
Streamlit interface for medical RAG system demonstration.
"""
import streamlit as st
import sys
from pathlib import Path
import time
import json
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import MedicalDataLoader
from retrieval import HybridRetriever
from generation import MedicalResponseGenerator
from eval import MedicalRAGEvaluator
import structlog

logger = structlog.get_logger()

class MedicalRAGApp:
    """Streamlit application for medical RAG system."""
    
    def __init__(self):
        self.loader = None
        self.retriever = None
        self.generator = None
        self.evaluator = None
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        if 'system_initialized' not in st.session_state:
            with st.spinner("Initializing Medical RAG System..."):
                try:
                    # Initialize components
                    self.loader = MedicalDataLoader()
                    self.retriever = HybridRetriever()
                    self.generator = MedicalResponseGenerator()
                    self.evaluator = MedicalRAGEvaluator()
                    
                    # Load data if not already done
                    if 'data_loaded' not in st.session_state:
                        self._load_sample_data()
                    
                    st.session_state.system_initialized = True
                    st.success("System initialized successfully!")
                    
                except Exception as e:
                    st.error(f"Failed to initialize system: {e}")
                    return
    
    def _load_sample_data(self):
        """Load sample medical data."""
        try:
            # Generate sample data
            icd_codes = self.loader.parse_icd10_codes(Path("data/icd10_processed.csv"))
            cpt_codes = self.loader.parse_cpt_codes(Path("data/cpt_processed.csv"))
            all_codes = icd_codes + cpt_codes
            chunks = self.loader.create_chunks_with_metadata(all_codes)
            
            # Initialize retriever with data
            self.retriever.initialize_vectorstore(chunks)
            
            st.session_state.data_loaded = True
            st.session_state.total_codes = len(all_codes)
            
        except Exception as e:
            st.error(f"Failed to load data: {e}")
    
    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="Medical RAG System",
            page_icon="🏥",
            layout="wide"
        )
        
        st.title("🏥 Medical RAG System for ICD-10/CPT Coding")
        st.markdown("Advanced retrieval-augmented generation system for automated medical code lookup")
        
        # Sidebar for system info
        with st.sidebar:
            st.header("System Information")
            
            if st.session_state.get('system_initialized', False):
                st.success("✅ System Ready")
                if st.session_state.get('data_loaded', False):
                    st.info(f"📊 {st.session_state.get('total_codes', 0)} medical codes loaded")
            else:
                st.warning("⏳ System Initializing...")
            
            st.markdown("---")
            st.header("System Components")
            st.markdown("""
            - **Data Sources**: ICD-10-CM & CPT codes
            - **Retrieval**: Hybrid (Semantic + BM25)
            - **Enhancement**: UMLS synonym expansion
            - **Generation**: GPT-4o-mini
            - **Evaluation**: RAGAS metrics
            """)
            
            # System settings
            st.markdown("---")
            st.header("Settings")
            specialty_filter = st.selectbox(
                "Specialty Filter",
                ["None", "cardiology", "gastroenterology", "neurology", "orthopedics", "radiology"]
            )
            
            include_rationale = st.checkbox("Include Medical Rationale", value=True)
            show_retrieved_docs = st.checkbox("Show Retrieved Documents", value=True)
        
        # Main interface
        tab1, tab2, tab3 = st.tabs(["💬 Query Interface", "📊 System Evaluation", "📋 Sample Queries"])
        
        with tab1:
            self._render_query_interface(specialty_filter, include_rationale, show_retrieved_docs)
        
        with tab2:
            self._render_evaluation_interface()
        
        with tab3:
            self._render_sample_queries()
    
    def _render_query_interface(self, specialty_filter, include_rationale, show_retrieved_docs):
        """Render the main query interface."""
        st.header("Medical Code Lookup")
        
        # Query input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Enter clinical scenario:",
                placeholder="e.g., patient with acute chest pain and dyspnea",
                help="Describe the clinical scenario for which you need medical codes"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        # Process query
        if search_button and query:
            if not st.session_state.get('system_initialized', False):
                st.error("System not initialized. Please wait for initialization to complete.")
                return
            
            # Query processing with timing
            start_time = time.time()
            
            with st.spinner("Processing query..."):
                try:
                    # Retrieve relevant documents
                    specialty = specialty_filter if specialty_filter != "None" else None
                    retrieved_docs = self.retriever.retrieve(query, specialty_filter=specialty)
                    
                    # Generate response
                    generation_result = self.generator.generate_response(query, retrieved_docs)
                    
                    response_time = time.time() - start_time
                    
                    # Display results
                    self._display_results(query, generation_result, retrieved_docs, response_time, show_retrieved_docs)
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
        
        # Example queries
        st.markdown("---")
        st.subheader("Example Queries")
        example_queries = [
            "patient with acute chest pain and dyspnea",
            "routine office visit for diabetes management", 
            "migraine headache treatment",
            "blood glucose test",
            "chronic kidney disease stage 5"
        ]
        
        cols = st.columns(len(example_queries))
        for i, example in enumerate(example_queries):
            with cols[i]:
                if st.button(f"Try: {example[:20]}...", key=f"example_{i}"):
                    st.session_state.example_query = example
                    st.rerun()
        
        # Handle example query selection
        if 'example_query' in st.session_state:
            st.session_state.query_input = st.session_state.example_query
            del st.session_state.example_query
    
    def _display_results(self, query, generation_result, retrieved_docs, response_time, show_retrieved_docs):
        """Display query results."""
        st.markdown("---")
        st.header("Results")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Response Time", f"{response_time:.2f}s")
        
        with col2:
            st.metric("Confidence", generation_result.get('confidence', 'Unknown'))
        
        with col3:
            reliability_score = generation_result.get('hallucination_score', 0.5)
            st.metric("Reliability", f"{reliability_score:.2f}")
        
        with col4:
            st.metric("Retrieved Docs", len(retrieved_docs))
        
        # Generated response
        st.subheader("Generated Response")
        
        # Check reliability
        is_reliable = generation_result.get('is_reliable', False)
        if is_reliable:
            st.success("✅ High reliability response")
        else:
            st.warning("⚠️ Response may require verification")
        
        st.markdown(generation_result.get('response', 'No response generated'))
        
        # Extracted codes
        extracted_codes = generation_result.get('codes', [])
        if extracted_codes:
            st.subheader("Extracted Medical Codes")
            
            codes_df = pd.DataFrame(extracted_codes)
            st.dataframe(codes_df, use_container_width=True)
        
        # Retrieved documents
        if show_retrieved_docs and retrieved_docs:
            st.subheader("Retrieved Documents")
            
            for i, doc in enumerate(retrieved_docs):
                with st.expander(f"Document {i+1}: {doc.metadata.get('code', 'N/A')}"):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown("**Metadata:**")
                        st.json(doc.metadata)
                    
                    with col2:
                        st.markdown("**Content:**")
                        st.markdown(doc.page_content)
    
    def _render_evaluation_interface(self):
        """Render system evaluation interface."""
        st.header("System Performance Evaluation")
        
        if st.button("Run Full Evaluation"):
            if not st.session_state.get('system_initialized', False):
                st.error("System not initialized")
                return
            
            with st.spinner("Running comprehensive evaluation..."):
                try:
                    # Create mock RAG pipeline for evaluation
                    class MockRAGPipeline:
                        def __init__(self, retriever, generator):
                            self.retriever = retriever
                            self.generator = generator
                        
                        def retrieve(self, query):
                            return self.retriever.retrieve(query)
                        
                        def generate(self, query, docs):
                            return self.generator.generate_response(query, docs)
                    
                    pipeline = MockRAGPipeline(self.retriever, self.generator)
                    metrics = self.evaluator.evaluate_system(pipeline)
                    
                    # Display results
                    st.subheader("Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Precision@5", f"{metrics.precision_at_5:.3f}", 
                                 delta=f"Target: 0.800")
                        st.metric("Accuracy", f"{metrics.accuracy:.3f}",
                                 delta=f"Target: 0.800")
                    
                    with col2:
                        st.metric("MRR", f"{metrics.mrr:.3f}",
                                 delta=f"Target: 0.700")
                        st.metric("NDCG@5", f"{metrics.ndcg_at_5:.3f}",
                                 delta=f"Target: 0.750")
                    
                    with col3:
                        st.metric("Avg Response Time", f"{metrics.avg_response_time:.2f}s",
                                 delta=f"Target: <1.0s")
                        st.metric("Hallucination Rate", f"{metrics.hallucination_rate:.1%}",
                                 delta=f"Target: <10%")
                    
                    # Performance report
                    st.subheader("Performance Assessment")
                    report = self.evaluator.generate_performance_report(metrics)
                    st.markdown(report)
                    
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
        
        # Historical results
        st.subheader("Historical Results")
        results_dir = Path("data/evaluation_results")
        
        if results_dir.exists():
            result_files = list(results_dir.glob("metrics_summary_*.csv"))
            if result_files:
                # Load and display historical results
                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_csv(latest_file)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No evaluation results found. Run evaluation to see results.")
        else:
            st.info("No evaluation results directory found.")
    
    def _render_sample_queries(self):
        """Render sample queries and explanations."""
        st.header("Sample Medical Scenarios")
        
        samples = [
            {
                "scenario": "Acute Chest Pain",
                "query": "patient with acute chest pain and dyspnea",
                "expected_codes": ["R07.9", "I20.0"],
                "explanation": "This scenario typically involves chest pain coding (R07.9) and may include cardiac conditions like unstable angina (I20.0)."
            },
            {
                "scenario": "Diabetes Management",
                "query": "routine office visit for diabetes management",
                "expected_codes": ["99214", "E11.9"],
                "explanation": "Routine diabetes visits use evaluation codes (99214) combined with diabetes diagnosis codes (E11.9)."
            },
            {
                "scenario": "Migraine Treatment",
                "query": "migraine headache treatment",
                "expected_codes": ["G43.909"],
                "explanation": "Migraine conditions are coded using specific G43 codes based on type and severity."
            },
            {
                "scenario": "Laboratory Test",
                "query": "blood glucose test",
                "expected_codes": ["82947"],
                "explanation": "Laboratory procedures use CPT codes, with glucose tests typically coded as 82947."
            }
        ]
        
        for sample in samples:
            with st.expander(f"📋 {sample['scenario']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Query:** {sample['query']}")
                    st.markdown(f"**Explanation:** {sample['explanation']}")
                
                with col2:
                    st.markdown("**Expected Codes:**")
                    for code in sample['expected_codes']:
                        st.code(code)
                
                if st.button(f"Test this scenario", key=f"test_{sample['scenario']}"):
                    # This would trigger the query processing
                    st.info("Click the search button in the Query Interface tab to test this scenario")

def main():
    """Main application entry point."""
    app = MedicalRAGApp()
    app.run()

if __name__ == "__main__":
    main()
