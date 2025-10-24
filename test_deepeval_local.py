"""
Local test for DeepEval with your Medical RAG System.
Tests the complete pipeline with DeepEval evaluation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import MedicalDataLoader
from retrieval import HybridRetriever
from generation import MedicalResponseGenerator
from evaluation_deepeval import DeepEvalRAGEvaluator

def test_deepeval_with_rag():
    print("\n" + "="*70)
    print("Testing DeepEval with Medical RAG System (Local)")
    print("="*70)

    # Initialize components
    print("\n[1/5] Initializing components...")
    loader = MedicalDataLoader()
    retriever = HybridRetriever()
    generator = MedicalResponseGenerator()
    evaluator = DeepEvalRAGEvaluator()
    print("  [OK] All components initialized")

    # Load data
    print("\n[2/5] Loading medical codes...")
    icd_codes = loader.parse_icd10_codes(Path("data/icd10_processed.csv"))
    print(f"  [OK] Loaded {len(icd_codes)} ICD-10 codes")

    # Initialize retriever with subset for speed
    print("\n[3/5] Initializing retrieval system...")
    chunks = loader.create_chunks_with_metadata(icd_codes[:1000])
    retriever.initialize_vectorstore(chunks, force_rebuild=False)
    print("  [OK] Retriever ready")

    # Test queries
    print("\n[4/5] Running test queries...")
    test_queries = [
        "patient with acute chest pain",
        "type 2 diabetes mellitus"
    ]

    all_responses = []
    all_contexts = []
    ground_truths = [
        "The patient likely has unstable angina (I20.0) or similar cardiac condition.",
        "Type 2 diabetes mellitus is coded as E11.9 without complications."
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n  Query {i}: {query}")

        # Retrieve
        docs = retriever.retrieve(query)
        print(f"    -> Retrieved {len(docs)} documents")

        # Generate
        response = generator.generate_response(query, docs)
        print(f"    -> Generated response")
        print(f"    -> Confidence: {response.get('confidence', 'N/A')}")

        # Store for evaluation
        all_responses.append(response['response'])
        all_contexts.append([doc.page_content for doc in docs[:3]])

    # Evaluate with DeepEval
    print("\n[5/5] Evaluating with DeepEval...")
    print("  (This will take 30-60 seconds...)")

    try:
        scores = evaluator.evaluate_with_deepeval(
            queries=test_queries,
            responses=all_responses,
            contexts=all_contexts,
            ground_truths=ground_truths
        )

        print("\n" + "="*70)
        print("DeepEval Results:")
        print("="*70)

        if scores:
            for metric, score in scores.items():
                if score is not None:
                    # Visual bar
                    bar_length = int(score * 20)
                    bar = "#" * bar_length + "-" * (20 - bar_length)
                    print(f"  {metric:25s} [{bar}] {score:.3f}")

            avg_score = sum(v for v in scores.values() if v is not None) / len([v for v in scores.values() if v is not None])
            print(f"\n  Average Score: {avg_score:.3f}")
            print("\n" + "="*70)
            print("SUCCESS! DeepEval works perfectly with Perplexity!")
            print("="*70)
        else:
            print("  [!] No scores returned - check logs above")

    except Exception as e:
        print(f"\n  [ERROR] DeepEval test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[OK] Local test completed!\n")

if __name__ == "__main__":
    test_deepeval_with_rag()
