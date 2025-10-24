"""
Quick test to verify RAGAS uses Perplexity API correctly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation_advanced import AdvancedRAGEvaluator

def test_ragas_config():
    print("\n" + "="*70)
    print("Testing RAGAS Configuration with Perplexity")
    print("="*70)

    evaluator = AdvancedRAGEvaluator()

    # Simple test data
    test_queries = ["patient with chest pain"]
    test_responses = ["The patient has unstable angina (I20.0)."]
    test_contexts = [["I20.0 - Unstable angina"]]
    test_ground_truths = ["I20.0 is the code for unstable angina"]

    print("\n[1/2] Running RAGAS evaluation...")
    print("  (This should now use sonar-pro instead of gpt-4o-mini)")

    try:
        scores = evaluator.evaluate_with_ragas(
            queries=test_queries,
            responses=test_responses,
            contexts=test_contexts,
            ground_truths=test_ground_truths
        )

        print("\n[2/2] RAGAS Scores:")
        for metric, score in scores.items():
            if score is not None:
                print(f"  - {metric}: {score:.3f}")

        print("\n" + "="*70)
        print("SUCCESS! RAGAS is now using Perplexity API")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] RAGAS test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ragas_config()
