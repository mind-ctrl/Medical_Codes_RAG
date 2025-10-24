"""
Advanced RAGAS-based evaluation for Medical RAG system.
Includes automated quality metrics, medical domain-specific scoring, and detailed analytics.
"""
from dotenv import load_dotenv
load_dotenv()

import structlog
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import json

logger = structlog.get_logger()

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # RAGAS metrics
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    # Custom medical metrics
    code_accuracy: float
    hallucination_rate: float
    avg_confidence: float

    # Performance metrics
    avg_latency: float
    cache_hit_rate: float

    # Overall score
    overall_score: float

class AdvancedRAGEvaluator:
    """Advanced evaluation with RAGAS and medical-specific metrics."""

    def __init__(self):
        self.logger = logger
        self.evaluation_history = []

    def evaluate_with_ragas(
        self,
        queries: List[str],
        responses: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate using RAGAS metrics.

        Args:
            queries: List of user queries
            responses: List of generated responses
            contexts: List of retrieved context documents
            ground_truths: Optional list of expected answers

        Returns:
            Dictionary of RAGAS scores
        """
        try:
            import os
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                context_entity_recall,
                answer_similarity,
                answer_correctness
            )
            from datasets import Dataset
            from langchain_openai import ChatOpenAI

            # Configure RAGAS to use Perplexity instead of OpenAI
            api_key = os.getenv("OPENAI_API_KEY", "")
            api_base = os.getenv("OPENAI_API_BASE", "https://api.perplexity.ai")
            model_name = os.getenv("LLM_MODEL", "sonar-pro")

            # Create LLM instance for RAGAS
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=api_base,
                temperature=0.0
            )

            self.logger.info(f"Configuring RAGAS with model: {model_name} at {api_base}")

            # Prepare dataset for RAGAS
            eval_data = {
                "question": queries,
                "answer": responses,
                "contexts": contexts,
            }

            if ground_truths:
                eval_data["ground_truth"] = ground_truths

            dataset = Dataset.from_dict(eval_data)

            # Select metrics based on available data
            metrics_to_use = [
                faithfulness,
                answer_relevancy,
            ]

            if ground_truths:
                metrics_to_use.extend([
                    context_precision,
                    context_recall,
                    answer_correctness,
                    answer_similarity
                ])

            # Run evaluation with custom LLM
            self.logger.info(f"Running RAGAS evaluation with {len(metrics_to_use)} metrics")
            results = evaluate(
                dataset,
                metrics=metrics_to_use,
                llm=llm,  # Use Perplexity instead of default OpenAI
                embeddings=None  # Will use default embeddings
            )

            # Convert to dict (RAGAS returns a Result object)
            try:
                # Try converting to pandas first (most reliable)
                if hasattr(results, 'to_pandas'):
                    df = results.to_pandas()
                    scores = {col: float(df[col].mean()) for col in df.columns}
                elif hasattr(results, '__dict__'):
                    # Access as dictionary
                    scores = dict(results)
                else:
                    # Fallback
                    scores = {}
                    for metric in ['faithfulness', 'answer_relevancy', 'context_precision',
                                   'context_recall', 'answer_correctness', 'answer_similarity']:
                        if hasattr(results, metric):
                            scores[metric] = getattr(results, metric)
            except:
                # Last resort: try dict access
                scores = dict(results) if hasattr(results, '__iter__') else {}

            self.logger.info(f"RAGAS evaluation completed: {scores}")
            return scores

        except Exception as e:
            self.logger.error(f"RAGAS evaluation failed: {e}")
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
            if response.get('hallucination_score', 1.0) < 0.6:
                total_hallucinations += 1

            # Collect confidence
            confidence = response.get('confidence', 'Medium')
            confidence_map = {'High': 1.0, 'Medium': 0.5, 'Low': 0.0}
            confidence_scores.append(confidence_map.get(confidence, 0.5))

            # Compare with ground truth if available
            if ground_truth_codes and i < len(ground_truth_codes):
                ground_truth = set(ground_truth_codes[i])
                extracted = set(extracted_codes)
                correct_codes += len(extracted.intersection(ground_truth))

        # Calculate metrics
        code_accuracy = (correct_codes / total_codes) if total_codes > 0 and ground_truth_codes else 0.0
        hallucination_rate = (total_hallucinations / len(responses)) if responses else 0.0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        return {
            "code_accuracy": code_accuracy,
            "hallucination_rate": hallucination_rate,
            "avg_confidence": avg_confidence,
            "total_codes_extracted": total_codes
        }

    def evaluate_full_pipeline(
        self,
        test_queries: List[str],
        rag_pipeline,
        ground_truth_codes: Optional[List[List[str]]] = None,
        ground_truth_answers: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """
        Comprehensive evaluation of the full RAG pipeline.

        Args:
            test_queries: List of test queries
            rag_pipeline: RAG pipeline object with retrieve() and generate() methods
            ground_truth_codes: Optional expected codes
            ground_truth_answers: Optional expected answers

        Returns:
            EvaluationMetrics object
        """
        self.logger.info(f"Starting full pipeline evaluation with {len(test_queries)} queries")

        # Run pipeline on all queries
        responses = []
        contexts_list = []
        answers = []
        latencies = []

        import time

        for query in test_queries:
            start_time = time.time()

            # Retrieve
            retrieved_docs = rag_pipeline.retrieve(query)
            contexts = [doc.page_content for doc in retrieved_docs]
            contexts_list.append(contexts)

            # Generate
            response = rag_pipeline.generate(query, retrieved_docs)
            responses.append(response)
            answers.append(response.get('response', ''))

            latency = time.time() - start_time
            latencies.append(latency)

        # RAGAS evaluation
        ragas_scores = self.evaluate_with_ragas(
            queries=test_queries,
            responses=answers,
            contexts=contexts_list,
            ground_truths=ground_truth_answers
        )

        # Medical accuracy evaluation
        medical_scores = self.evaluate_medical_accuracy(
            responses=responses,
            ground_truth_codes=ground_truth_codes
        )

        # Calculate overall metrics
        metrics = EvaluationMetrics(
            faithfulness=ragas_scores.get('faithfulness', 0.0),
            answer_relevancy=ragas_scores.get('answer_relevancy', 0.0),
            context_precision=ragas_scores.get('context_precision') or 0.0,
            context_recall=ragas_scores.get('context_recall') or 0.0,
            code_accuracy=medical_scores['code_accuracy'],
            hallucination_rate=medical_scores['hallucination_rate'],
            avg_confidence=medical_scores['avg_confidence'],
            avg_latency=np.mean(latencies),
            cache_hit_rate=0.0,  # TODO: Get from cache
            overall_score=self._calculate_overall_score(ragas_scores, medical_scores)
        )

        # Save evaluation
        self.evaluation_history.append({
            "timestamp": pd.Timestamp.now(),
            "metrics": metrics,
            "num_queries": len(test_queries)
        })

        self.logger.info(f"Evaluation completed. Overall score: {metrics.overall_score:.3f}")

        return metrics

    def _calculate_overall_score(
        self,
        ragas_scores: Dict[str, Any],
        medical_scores: Dict[str, float]
    ) -> float:
        """Calculate weighted overall score."""
        # Weights
        weights = {
            'faithfulness': 0.25,
            'answer_relevancy': 0.20,
            'code_accuracy': 0.30,
            'avg_confidence': 0.15,
            'hallucination_penalty': 0.10
        }

        score = 0.0
        score += ragas_scores.get('faithfulness', 0.0) * weights['faithfulness']
        score += ragas_scores.get('answer_relevancy', 0.0) * weights['answer_relevancy']
        score += medical_scores['code_accuracy'] * weights['code_accuracy']
        score += medical_scores['avg_confidence'] * weights['avg_confidence']
        score -= medical_scores['hallucination_rate'] * weights['hallucination_penalty']

        return max(0.0, min(1.0, score))

    def generate_report(self, metrics: EvaluationMetrics) -> str:
        """Generate a detailed evaluation report."""
        report = f"""
================================================================
           Medical RAG System Evaluation Report
================================================================

RAGAS Metrics:
  - Faithfulness:        {metrics.faithfulness:.3f} (groundedness to sources)
  - Answer Relevancy:    {metrics.answer_relevancy:.3f} (relevance to query)
  - Context Precision:   {metrics.context_precision:.3f} (retrieval accuracy)
  - Context Recall:      {metrics.context_recall:.3f} (retrieval completeness)

Medical Domain Metrics:
  - Code Accuracy:       {metrics.code_accuracy:.3f} (correct code extraction)
  - Hallucination Rate:  {metrics.hallucination_rate:.3f} (lower is better)
  - Avg Confidence:      {metrics.avg_confidence:.3f} (system confidence)

Performance Metrics:
  - Avg Latency:         {metrics.avg_latency:.2f}s (response time)
  - Cache Hit Rate:      {metrics.cache_hit_rate:.1%} (cache efficiency)

Overall Score:           {metrics.overall_score:.3f} / 1.000

Rating: {self._get_rating(metrics.overall_score)}
        """
        return report.strip()

    def _get_rating(self, score: float) -> str:
        """Convert score to rating."""
        if score >= 0.9:
            return "***** Excellent"
        elif score >= 0.8:
            return "**** Very Good"
        elif score >= 0.7:
            return "*** Good"
        elif score >= 0.6:
            return "** Fair"
        else:
            return "* Needs Improvement"

    def save_evaluation(self, metrics: EvaluationMetrics, output_path: Path):
        """Save evaluation results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        eval_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": {
                "faithfulness": metrics.faithfulness,
                "answer_relevancy": metrics.answer_relevancy,
                "context_precision": metrics.context_precision,
                "context_recall": metrics.context_recall,
                "code_accuracy": metrics.code_accuracy,
                "hallucination_rate": metrics.hallucination_rate,
                "avg_confidence": metrics.avg_confidence,
                "avg_latency": metrics.avg_latency,
                "cache_hit_rate": metrics.cache_hit_rate,
                "overall_score": metrics.overall_score
            }
        }

        with open(output_path, 'w') as f:
            json.dump(eval_data, f, indent=2)

        self.logger.info(f"Evaluation saved to {output_path}")

def main():
    """Demo evaluation."""
    evaluator = AdvancedRAGEvaluator()

    # Example data
    test_queries = [
        "patient with acute chest pain and dyspnea",
        "routine office visit for diabetes management"
    ]

    # Mock pipeline
    class MockPipeline:
        def retrieve(self, query):
            from langchain_core.documents import Document
            return [Document(page_content=f"Mock doc for {query}", metadata={"code": "I20.0"})]

        def generate(self, query, docs):
            return {
                "response": f"Mock response for {query}",
                "codes": [{"code": "I20.0", "description": "Unstable angina"}],
                "confidence": "High",
                "hallucination_score": 0.85
            }

    pipeline = MockPipeline()

    metrics = evaluator.evaluate_full_pipeline(
        test_queries=test_queries,
        rag_pipeline=pipeline
    )

    print(evaluator.generate_report(metrics))

if __name__ == "__main__":
    main()
