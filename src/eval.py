"""
Comprehensive evaluation framework for medical RAG system using RAGAS and custom metrics.
"""
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json
import structlog
from dataclasses import dataclass, asdict
from sklearn.metrics import precision_score, recall_score, f1_score
import asyncio

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision, 
        context_recall, 
        answer_relevancy, 
        faithfulness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    logger.warning("RAGAS not available. Using custom metrics only.")
    RAGAS_AVAILABLE = False

logger = structlog.get_logger()

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    precision_at_5: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_5: float = 0.0  # Normalized Discounted Cumulative Gain
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    accuracy: float = 0.0
    avg_response_time: float = 0.0
    hallucination_rate: float = 0.0

@dataclass
class EvaluationExample:
    """Single evaluation example."""
    query: str
    ground_truth_codes: List[str]
    retrieved_codes: List[str]
    generated_response: str
    context: List[str]
    relevance_scores: List[int] = None  # 1 for relevant, 0 for not relevant
    response_time: float = 0.0
    hallucination_score: float = 0.0

class MedicalRAGEvaluator:
    """Comprehensive evaluator for medical RAG systems."""
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.gold_dataset_path = data_dir / "gold_set.json"
        self.results_dir = data_dir / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize evaluation dataset
        self.gold_dataset = self._load_or_create_gold_dataset()
    
    def _load_or_create_gold_dataset(self) -> List[Dict[str, Any]]:
        """Load existing gold dataset or create synthetic one."""
        if self.gold_dataset_path.exists():
            logger.info(f"Loading gold dataset from {self.gold_dataset_path}")
            with open(self.gold_dataset_path, 'r') as f:
                return json.load(f)
        else:
            logger.info("Creating synthetic gold dataset")
            return self._create_synthetic_gold_dataset()
    
    def _create_synthetic_gold_dataset(self) -> List[Dict[str, Any]]:
        """Create synthetic evaluation dataset."""
        synthetic_examples = [
            {
                "query": "patient with acute chest pain and dyspnea",
                "ground_truth_codes": ["R07.9", "I20.0"],
                "ground_truth_answer": "The patient presents with acute chest pain and dyspnea. Appropriate codes include R07.9 for chest pain, unspecified, and consider I20.0 for unstable angina if cardiac cause is suspected.",
                "context": [
                    "R07.9: Chest pain, unspecified - used for chest pain when specific cause is not determined",
                    "I20.0: Unstable angina - cardiac chest pain due to coronary artery disease",
                    "R06.0: Dyspnea - difficulty breathing or shortness of breath"
                ]
            },
            {
                "query": "routine office visit for diabetes management",
                "ground_truth_codes": ["99214", "E11.9"],
                "ground_truth_answer": "For routine diabetes management visit, use 99214 for established patient office visit and E11.9 for Type 2 diabetes mellitus without complications.",
                "context": [
                    "99214: Office visit for established patient, detailed history and examination",
                    "E11.9: Type 2 diabetes mellitus without complications",
                    "99213: Office visit for established patient, less complex"
                ]
            },
            {
                "query": "migraine headache treatment",
                "ground_truth_codes": ["G43.909"],
                "ground_truth_answer": "Migraine headache should be coded as G43.909 for migraine, unspecified, not intractable, without status migrainosus.",
                "context": [
                    "G43.909: Migraine, unspecified, not intractable, without status migrainosus",
                    "G43.919: Migraine, unspecified, intractable, without status migrainosus",
                    "R51: Headache - used for non-specific headaches"
                ]
            },
            {
                "query": "chronic kidney disease stage 5",
                "ground_truth_codes": ["N18.6"],
                "ground_truth_answer": "Chronic kidney disease stage 5 (end stage renal disease) should be coded as N18.6.",
                "context": [
                    "N18.6: End stage renal disease - stage 5 chronic kidney disease",
                    "N18.5: Chronic kidney disease, stage 5 (severe)",
                    "N18.4: Chronic kidney disease, stage 4 (severe)"
                ]
            },
            {
                "query": "blood glucose test",
                "ground_truth_codes": ["82947"],
                "ground_truth_answer": "Blood glucose test should be coded with CPT code 82947 for glucose; quantitative, blood (except reagent strip).",
                "context": [
                    "82947: Glucose; quantitative, blood (except reagent strip)",
                    "82948: Glucose; blood, reagent strip",
                    "82950: Glucose; post glucose dose (includes glucose)"
                ]
            }
        ]
        
        # Save synthetic dataset
        with open(self.gold_dataset_path, 'w') as f:
            json.dump(synthetic_examples, f, indent=2)
        
        logger.info(f"Created synthetic gold dataset with {len(synthetic_examples)} examples")
        return synthetic_examples
    
    def evaluate_retrieval(self, examples: List[EvaluationExample]) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        if not examples:
            return {}
        
        # Calculate Precision@5
        precision_scores = []
        for example in examples:
            if example.relevance_scores:
                precision = sum(example.relevance_scores[:5]) / min(5, len(example.relevance_scores))
                precision_scores.append(precision)
        
        precision_at_5 = np.mean(precision_scores) if precision_scores else 0.0
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr_scores = []
        for example in examples:
            reciprocal_rank = 0.0
            for i, code in enumerate(example.retrieved_codes):
                if code in example.ground_truth_codes:
                    reciprocal_rank = 1.0 / (i + 1)
                    break
            mrr_scores.append(reciprocal_rank)
        
        mrr = np.mean(mrr_scores)
        
        # Calculate NDCG@5
        ndcg_scores = []
        for example in examples:
            dcg = 0.0
            idcg = 0.0
            
            # Calculate DCG
            for i, code in enumerate(example.retrieved_codes[:5]):
                relevance = 1 if code in example.ground_truth_codes else 0
                dcg += relevance / np.log2(i + 2)
            
            # Calculate IDCG (perfect ranking)
            sorted_relevance = sorted([1 if code in example.ground_truth_codes else 0 
                                     for code in example.retrieved_codes[:5]], reverse=True)
            for i, relevance in enumerate(sorted_relevance):
                idcg += relevance / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        ndcg_at_5 = np.mean(ndcg_scores)
        
        return {
            "precision_at_5": precision_at_5,
            "mrr": mrr,
            "ndcg_at_5": ndcg_at_5
        }
    
    def evaluate_with_ragas(self, examples: List[EvaluationExample]) -> Dict[str, float]:
        """Evaluate using RAGAS metrics."""
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS not available, skipping RAGAS evaluation")
            return {}
        
        # Prepare data for RAGAS
        eval_data = {
            'question': [],
            'contexts': [],
            'answer': [],
            'ground_truths': []
        }
        
        for example in examples:
            eval_data['question'].append(example.query)
            eval_data['contexts'].append(example.context)
            eval_data['answer'].append(example.generated_response)
            # RAGAS expects ground truth as list of strings
            eval_data['ground_truths'].append([example.generated_response])  # Simplified
        
        try:
            dataset = Dataset.from_dict(eval_data)
            
            # Run RAGAS evaluation
            result = evaluate(
                dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    answer_relevancy,
                    faithfulness
                ]
            )
            
            return {
                "context_precision": result['context_precision'],
                "context_recall": result['context_recall'],
                "answer_relevancy": result['answer_relevancy'],
                "faithfulness": result['faithfulness']
            }
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}
    
    def evaluate_system(self, rag_pipeline, save_results: bool = True) -> EvaluationMetrics:
        """Comprehensive system evaluation."""
        logger.info("Starting comprehensive system evaluation")
        
        evaluation_examples = []
        total_response_time = 0.0
        
        # Run evaluation on gold dataset
        for i, gold_example in enumerate(self.gold_dataset):
            logger.info(f"Evaluating example {i+1}/{len(self.gold_dataset)}")
            
            query = gold_example["query"]
            ground_truth_codes = gold_example["ground_truth_codes"]
            
            # Time the pipeline
            import time
            start_time = time.time()
            
            try:
                # Run RAG pipeline
                retrieved_docs = rag_pipeline.retrieve(query)
                generated_result = rag_pipeline.generate(query, retrieved_docs)
                
                response_time = time.time() - start_time
                total_response_time += response_time
                
                # Extract information
                retrieved_codes = [doc.metadata.get('code', '') for doc in retrieved_docs]
                generated_response = generated_result.get('response', '')
                context = [doc.page_content for doc in retrieved_docs]
                hallucination_score = generated_result.get('hallucination_score', 0.5)
                
                # Calculate relevance scores
                relevance_scores = [1 if code in ground_truth_codes else 0 for code in retrieved_codes]
                
                example = EvaluationExample(
                    query=query,
                    ground_truth_codes=ground_truth_codes,
                    retrieved_codes=retrieved_codes,
                    generated_response=generated_response,
                    context=context,
                    relevance_scores=relevance_scores,
                    response_time=response_time,
                    hallucination_score=hallucination_score
                )
                
                evaluation_examples.append(example)
                
            except Exception as e:
                logger.error(f"Failed to evaluate example {i}: {e}")
                continue
        
        # Calculate metrics
        retrieval_metrics = self.evaluate_retrieval(evaluation_examples)
        ragas_metrics = self.evaluate_with_ragas(evaluation_examples)
        
        # Calculate additional metrics
        avg_response_time = total_response_time / len(evaluation_examples) if evaluation_examples else 0.0
        hallucination_rate = np.mean([1 - ex.hallucination_score for ex in evaluation_examples])
        
        # Calculate accuracy (percentage of queries with at least one correct code retrieved)
        accuracy_scores = []
        for example in evaluation_examples:
            has_correct_code = any(code in example.ground_truth_codes for code in example.retrieved_codes)
            accuracy_scores.append(1 if has_correct_code else 0)
        accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
        
        # Combine all metrics
        final_metrics = EvaluationMetrics(
            precision_at_5=retrieval_metrics.get('precision_at_5', 0.0),
            mrr=retrieval_metrics.get('mrr', 0.0),
            ndcg_at_5=retrieval_metrics.get('ndcg_at_5', 0.0),
            context_precision=ragas_metrics.get('context_precision', 0.0),
            context_recall=ragas_metrics.get('context_recall', 0.0),
            answer_relevancy=ragas_metrics.get('answer_relevancy', 0.0),
            faithfulness=ragas_metrics.get('faithfulness', 0.0),
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            hallucination_rate=hallucination_rate
        )
        
        if save_results:
            self._save_evaluation_results(final_metrics, evaluation_examples)
        
        logger.info(f"Evaluation completed. Precision@5: {final_metrics.precision_at_5:.3f}, "
                   f"MRR: {final_metrics.mrr:.3f}, Accuracy: {final_metrics.accuracy:.3f}")
        
        return final_metrics
    
    def _save_evaluation_results(self, metrics: EvaluationMetrics, examples: List[EvaluationExample]):
        """Save evaluation results to files."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics summary
        metrics_file = self.results_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # Save detailed results
        detailed_results = []
        for example in examples:
            detailed_results.append({
                "query": example.query,
                "ground_truth_codes": example.ground_truth_codes,
                "retrieved_codes": example.retrieved_codes,
                "relevance_scores": example.relevance_scores,
                "response_time": example.response_time,
                "hallucination_score": example.hallucination_score,
                "response_preview": example.generated_response[:200] + "..." if len(example.generated_response) > 200 else example.generated_response
            })
        
        detailed_file = self.results_dir / f"detailed_results_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Create CSV summary
        df = pd.DataFrame([asdict(metrics)])
        csv_file = self.results_dir / f"metrics_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Evaluation results saved to {self.results_dir}")
    
    def generate_performance_report(self, metrics: EvaluationMetrics) -> str:
        """Generate human-readable performance report."""
        report = f"""
# Medical RAG System Performance Report

## Overall Performance
- **Accuracy**: {metrics.accuracy:.1%} (Target: >80%)
- **Precision@5**: {metrics.precision_at_5:.3f} (Target: >0.80)
- **Mean Reciprocal Rank**: {metrics.mrr:.3f} (Target: >0.70)
- **NDCG@5**: {metrics.ndcg_at_5:.3f} (Target: >0.75)

## Response Quality (RAGAS Metrics)
- **Context Precision**: {metrics.context_precision:.3f}
- **Context Recall**: {metrics.context_recall:.3f}
- **Answer Relevancy**: {metrics.answer_relevancy:.3f}
- **Faithfulness**: {metrics.faithfulness:.3f}

## System Performance
- **Average Response Time**: {metrics.avg_response_time:.2f}s (Target: <1.0s)
- **Hallucination Rate**: {metrics.hallucination_rate:.1%} (Target: <10%)

## Assessment
"""
        
        # Performance assessment
        if metrics.precision_at_5 >= 0.80:
            report += "✅ **EXCELLENT**: Precision meets production standards\n"
        elif metrics.precision_at_5 >= 0.70:
            report += "⚠️ **GOOD**: Precision acceptable but room for improvement\n"
        else:
            report += "❌ **NEEDS IMPROVEMENT**: Precision below acceptable threshold\n"
        
        if metrics.avg_response_time <= 1.0:
            report += "✅ **EXCELLENT**: Response time meets real-time requirements\n"
        else:
            report += "⚠️ **ATTENTION**: Response time may impact user experience\n"
        
        return report

def main():
    """Demo evaluation system."""
    # This would typically be called with a real RAG pipeline
    evaluator = MedicalRAGEvaluator()
    
    # Mock evaluation results for demo
    mock_metrics = EvaluationMetrics(
        precision_at_5=0.85,
        mrr=0.78,
        ndcg_at_5=0.82,
        accuracy=0.90,
        avg_response_time=0.45,
        hallucination_rate=0.08
    )
    
    report = evaluator.generate_performance_report(mock_metrics)
    print(report)

if __name__ == "__main__":
    main()
