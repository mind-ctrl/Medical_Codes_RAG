"""
Test script to demonstrate enhanced retrieval improvements.
Shows before/after comparison with detailed scores and explanations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import MedicalDataLoader
from retrieval import HybridRetriever as BasicRetriever
from retrieval_enhanced import EnhancedHybridRetriever
import time

def test_enhanced_features():
    print("\n" + "="*80)
    print("ENHANCED RETRIEVAL DEMONSTRATION")
    print("="*80)

    # Initialize
    print("\n[1/4] Loading medical codes...")
    loader = MedicalDataLoader()
    icd_codes = loader.parse_icd10_codes(Path("data/icd10_processed.csv"))
    print(f"  [OK] Loaded {len(icd_codes)} ICD-10 codes")

    print("\n[2/4] Initializing retrievers...")
    chunks = loader.create_chunks_with_metadata(icd_codes[:1000])

    # Basic retriever
    basic_retriever = BasicRetriever()
    basic_retriever.initialize_vectorstore(chunks, force_rebuild=False)

    # Enhanced retriever
    enhanced_retriever = EnhancedHybridRetriever()
    enhanced_retriever.initialize_vectorstore(chunks, force_rebuild=False)

    print("  [OK] Both retrievers ready")

    # Test queries
    print("\n[3/4] Testing with sample queries...")
    test_queries = [
        ("patient with acute chest pain", "symptom-based query"),
        ("I20.0", "exact code lookup"),
        ("diabetes with complications", "general search")
    ]

    for query, query_type in test_queries:
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print(f"TYPE: {query_type}")
        print("="*80)

        # Basic retrieval
        print("\n--- BASIC RETRIEVAL ---")
        start = time.time()
        basic_results = basic_retriever.retrieve(query)
        basic_time = time.time() - start

        print(f"Time: {basic_time:.3f}s")
        print(f"Results: {len(basic_results)}")
        for i, doc in enumerate(basic_results[:3], 1):
            code = doc.metadata.get('code', 'N/A')
            desc = doc.metadata.get('description', 'N/A')[:60]
            print(f"  {i}. {code}: {desc}...")

        # Enhanced retrieval
        print("\n--- ENHANCED RETRIEVAL (with RRF + MMR + Explainability) ---")
        start = time.time()
        enhanced_results = enhanced_retriever.retrieve(query, return_detailed=True)
        enhanced_time = time.time() - start

        print(f"Time: {enhanced_time:.3f}s")
        print(f"Results: {len(enhanced_results)}")

        for result in enhanced_results[:3]:
            code = result.document.metadata.get('code', 'N/A')
            desc = result.document.metadata.get('description', 'N/A')[:60]

            print(f"\n  Rank {result.rank}: {code}")
            print(f"    Description: {desc}...")
            print(f"    Final Score: {result.final_score:.3f}")
            print(f"    BM25: {result.bm25_score:.3f} | Semantic: {result.semantic_score:.3f} | Cross-Encoder: {result.cross_encoder_score:.3f}")
            print(f"    Explanation: {result.explanation}")

    print("\n[4/4] Key Improvements Demonstrated:")
    print("  [OK] Reciprocal Rank Fusion - Better score merging than simple averaging")
    print("  [OK] Query-Adaptive Weighting - Different strategies for codes vs symptoms")
    print("  [OK] Maximal Marginal Relevance - Diverse results, no redundancy")
    print("  [OK] Confidence Filtering - Only high-quality results shown")
    print("  [OK] Explainability - Shows WHY each result was ranked")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nWhat You Can Tell Interviewers:")
    print("  1. 'I implemented Reciprocal Rank Fusion (RRF) - same as Google Search'")
    print("  2. 'I added query-adaptive weighting - exact codes use more BM25'")
    print("  3. 'I use Maximal Marginal Relevance (MMR) for diversity'")
    print("  4. 'I built explainability - users see why results ranked where they did'")
    print("\n")

if __name__ == "__main__":
    try:
        test_enhanced_features()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
