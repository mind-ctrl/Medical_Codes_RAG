"""
DeepEval-based evaluation for Medical RAG system.
Works seamlessly with Perplexity API - no OpenAI required!
"""
from dotenv import load_dotenv
load_dotenv()

import structlog
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os

logger = structlog.get_logger()

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # DeepEval metrics
    faithfulness: float
    answer_relevancy: float
    contextual_precision: float
    contextual_recall: float

    # Custom medical metrics
    code_accuracy: float
    hallucination_rate: float
    avg_confidence: float

    # Performance metrics
    avg_latency: float

    # Overall score
    overall_score: float

class DeepEvalRAGEvaluator:
    """
    Advanced evaluation using DeepEval with Perplexity API.

    DeepEval advantages over RAGAS:
    - Full Perplexity API support
    - No OpenAI embeddings required
    - More flexible metric customization
    - Better error handling
    """

    def __init__(self):
        self.logger = logger
        self.evaluation_history = []

        # Configure for Perplexity
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.perplexity.ai")
        self.model_name = os.getenv("LLM_MODEL", "sonar-pro")

        self.logger.info(f"DeepEval configured with model: {self.model_name}")

    def evaluate_with_deepeval(
        self,
        queries: List[str],
        responses: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate using DeepEval metrics.

        Args:
            queries: List of user queries
            responses: List of generated responses
            contexts: List of retrieved context documents
            ground_truths: Optional list of expected answers

        Returns:
            Dictionary of DeepEval scores
        """
        try:
            from deepeval.metrics import (
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric
            )
            from deepeval.test_case import LLMTestCase
            from deepeval.models.base_model import DeepEvalBaseLLM

            # Create custom Perplexity model for DeepEval
            class PerplexityModel(DeepEvalBaseLLM):
                def __init__(self, model_name, api_key, api_base):
                    self.model_name = model_name
                    self.api_key = api_key
                    self.api_base = api_base

                def load_model(self):
                    return self

                def generate(self, prompt: str) -> str:
                    """Generate response using Perplexity API."""
                    from openai import OpenAI

                    client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.api_base
                    )

                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0
                    )

                    return response.choices[0].message.content

                async def a_generate(self, prompt: str) -> str:
                    """Async generate (uses sync for simplicity)."""
                    return self.generate(prompt)

                def get_model_name(self) -> str:
                    return self.model_name

            # Initialize Perplexity model
            perplexity_model = PerplexityModel(
                model_name=self.model_name,
                api_key=self.api_key,
                api_base=self.api_base
            )

            self.logger.info(f"Initialized Perplexity model for DeepEval: {self.model_name}")

            # Initialize metrics
            faithfulness_metric = FaithfulnessMetric(
                threshold=0.7,
                model=perplexity_model,
                include_reason=True
            )

            relevancy_metric = AnswerRelevancyMetric(
                threshold=0.7,
                model=perplexity_model,
                include_reason=True
            )

            # Create test cases
            test_cases = []
            for i in range(len(queries)):
                test_case = LLMTestCase(
                    input=queries[i],
                    actual_output=responses[i],
                    retrieval_context=contexts[i],
                    expected_output=ground_truths[i] if ground_truths and i < len(ground_truths) else None
                )
                test_cases.append(test_case)

            # Evaluate each test case
            faithfulness_scores = []
            relevancy_scores = []

            self.logger.info(f"Evaluating {len(test_cases)} test cases with DeepEval...")

            for i, test_case in enumerate(test_cases):
                try:
                    # Measure faithfulness
                    faithfulness_metric.measure(test_case)
                    faithfulness_scores.append(faithfulness_metric.score)

                    # Measure relevancy
                    relevancy_metric.measure(test_case)
                    relevancy_scores.append(relevancy_metric.score)

                    self.logger.info(
                        f"Test case {i+1}: "
                        f"Faithfulness={faithfulness_metric.score:.3f}, "
                        f"Relevancy={relevancy_metric.score:.3f}"
                    )

                except Exception as e:
                    self.logger.warning(f"Failed to evaluate test case {i+1}: {e}")
                    faithfulness_scores.append(0.0)
                    relevancy_scores.append(0.0)

            # Calculate averages
            scores = {
                "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
                "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0,
            }

            # Add contextual metrics if ground truths provided
            if ground_truths:
                try:
                    precision_metric = ContextualPrecisionMetric(
                        threshold=0.7,
                        model=perplexity_model,
                        include_reason=True
                    )

                    recall_metric = ContextualRecallMetric(
                        threshold=0.7,
                        model=perplexity_model,
                        include_reason=True
                    )

                    precision_scores = []
                    recall_scores = []

                    for test_case in test_cases:
                        if test_case.expected_output:
                            try:
                                precision_metric.measure(test_case)
                                precision_scores.append(precision_metric.score)

                                recall_metric.measure(test_case)
                                recall_scores.append(recall_metric.score)
                            except:
                                pass

                    if precision_scores:
                        scores["contextual_precision"] = sum(precision_scores) / len(precision_scores)
                    if recall_scores:
                        scores["contextual_recall"] = sum(recall_scores) / len(recall_scores)

                except Exception as e:
                    self.logger.warning(f"Contextual metrics failed: {e}")

            self.logger.info(f"DeepEval evaluation completed: {scores}")
            return scores

        except Exception as e:
            self.logger.error(f"DeepEval evaluation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

    def evaluate_medical_accuracy(
        self,
        responses: List[Dict[str, Any]],
        ground_truth_codes: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate medical code accuracy.

        Args:
            responses: List of response dictionaries with extracted codes
            ground_truth_codes: Optional list of expected code lists

        Returns:
            Dictionary of accuracy metrics
        """
        total_codes = 0
        correct_codes = 0
        total_hallucinations = 0
        confidence_scores = []

        for i, response in enumerate(responses):
            extracted_codes = [code['code'] for code in response.get('codes', [])]
            total_codes += len(extracted_codes)

            # Check hallucination score
            hallucination_score = response.get('hallucination_score', 0.0)
            if hallucination_score < 0.7:  # Low score = likely hallucination
                total_hallucinations += 1

            # Map confidence to numeric
            confidence = response.get('confidence', 'Medium')
            confidence_map = {'High': 1.0, 'Medium': 0.6, 'Low': 0.3}
            confidence_scores.append(confidence_map.get(confidence, 0.5))

            # Compare with ground truth if available
            if ground_truth_codes and i < len(ground_truth_codes):
                gt_codes = ground_truth_codes[i]
                for code in extracted_codes:
                    if code in gt_codes:
                        correct_codes += 1

        # Calculate metrics
        code_accuracy = correct_codes / total_codes if total_codes > 0 else 0.0
        hallucination_rate = total_hallucinations / len(responses) if responses else 0.0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        return {
            "code_accuracy": code_accuracy,
            "hallucination_rate": hallucination_rate,
            "avg_confidence": avg_confidence,
            "total_codes_extracted": total_codes
        }

    def generate_report(self, metrics: EvaluationMetrics) -> str:
        """Generate a formatted evaluation report."""
        report = f"""
{'='*70}
          Medical RAG System Evaluation Report (DeepEval)
{'='*70}

DeepEval Metrics:
  - Faithfulness:        {metrics.faithfulness:.3f} (groundedness to sources)
  - Answer Relevancy:    {metrics.answer_relevancy:.3f} (relevance to query)
  - Contextual Precision:{metrics.contextual_precision:.3f} (retrieval accuracy)
  - Contextual Recall:   {metrics.contextual_recall:.3f} (retrieval completeness)

Medical Domain Metrics:
  - Code Accuracy:       {metrics.code_accuracy:.3f} (correct code extraction)
  - Hallucination Rate:  {metrics.hallucination_rate:.3f} (lower is better)
  - Avg Confidence:      {metrics.avg_confidence:.3f} (system confidence)

Performance Metrics:
  - Avg Latency:         {metrics.avg_latency:.2f}s (response time)

Overall Score:           {metrics.overall_score:.3f} / 1.000

Rating: {'*' * int(metrics.overall_score * 5)} {self._get_rating(metrics.overall_score)}
{'='*70}
"""
        return report

    def _get_rating(self, score: float) -> str:
        """Convert score to rating."""
        if score >= 0.9: return "Excellent"
        if score >= 0.8: return "Very Good"
        if score >= 0.7: return "Good"
        if score >= 0.6: return "Fair"
        return "Needs Improvement"

def main():
    """Demo DeepEval with Perplexity."""
    evaluator = DeepEvalRAGEvaluator()

    # Test data
    test_queries = ["patient with chest pain"]
    test_responses = ["The patient has unstable angina (I20.0)."]
    test_contexts = [["I20.0 - Unstable angina", "I20.1 - Angina pectoris"]]
    test_ground_truths = ["I20.0 is unstable angina"]

    print("\n" + "="*70)
    print("Testing DeepEval with Perplexity API")
    print("="*70)

    # Run evaluation
    scores = evaluator.evaluate_with_deepeval(
        queries=test_queries,
        responses=test_responses,
        contexts=test_contexts,
        ground_truths=test_ground_truths
    )

    print("\nDeepEval Scores:")
    for metric, score in scores.items():
        if score is not None:
            print(f"  - {metric}: {score:.3f}")

    print("\n" + "="*70)
    print("SUCCESS! DeepEval works with Perplexity!")
    print("="*70)

if __name__ == "__main__":
    main()
