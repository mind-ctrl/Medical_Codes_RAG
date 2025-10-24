"""
Medical Domain Embedding Fine-tuning Module.
Fine-tunes sentence-transformers on medical coding task for improved retrieval.
"""
from dotenv import load_dotenv
load_dotenv()

import structlog
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation
)
from torch.utils.data import DataLoader
import json

logger = structlog.get_logger()

class MedicalEmbeddingFinetuner:
    """Fine-tune embeddings for medical coding domain."""

    def __init__(
        self,
        base_model: str = 'all-MiniLM-L6-v2',
        output_path: str = './models/medical_embeddings'
    ):
        """
        Initialize fine-tuner.

        Args:
            base_model: Base sentence-transformer model
            output_path: Where to save fine-tuned model
        """
        self.base_model = base_model
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model = None
        logger.info(f"Initialized fine-tuner with base model: {base_model}")

    def create_training_data_from_codes(
        self,
        icd_codes: List[Dict[str, Any]],
        num_samples: int = 5000
    ) -> List[InputExample]:
        """
        Create training pairs from ICD-10 codes.

        Strategy:
        - Positive pairs: Query descriptions with their codes
        - Hard negatives: Similar but incorrect codes
        - Easy negatives: Random unrelated codes

        Args:
            icd_codes: List of ICD code dictionaries
            num_samples: Number of training samples to generate

        Returns:
            List of InputExample for training
        """
        logger.info(f"Creating {num_samples} training examples from {len(icd_codes)} codes")

        training_examples = []

        # Sample codes
        sampled_codes = np.random.choice(icd_codes, min(num_samples, len(icd_codes)), replace=False)

        for code_obj in sampled_codes:
            code = code_obj.code
            description = code_obj.description

            # Create query variations
            query_templates = [
                f"patient with {description.lower()}",
                f"diagnosis of {description.lower()}",
                f"{description.lower()} symptoms",
                description.lower()
            ]

            for query in query_templates[:2]:  # Use 2 variations per code
                # Positive example (query -> correct code description)
                training_examples.append(
                    InputExample(
                        texts=[query, description],
                        label=1.0  # High similarity
                    )
                )

                # Hard negative (similar code from same category)
                category = code[:3]  # First 3 chars of ICD-10
                similar_codes = [c for c in icd_codes if c.code.startswith(category) and c.code != code]
                if similar_codes:
                    hard_neg = np.random.choice(similar_codes)
                    training_examples.append(
                        InputExample(
                            texts=[query, hard_neg.description],
                            label=0.3  # Low-medium similarity
                        )
                    )

                # Easy negative (random unrelated code)
                easy_neg = np.random.choice(icd_codes)
                while easy_neg.code[:3] == category:
                    easy_neg = np.random.choice(icd_codes)

                training_examples.append(
                    InputExample(
                        texts=[query, easy_neg.description],
                        label=0.0  # No similarity
                    )
                )

        logger.info(f"Created {len(training_examples)} training examples")
        return training_examples

    def create_evaluation_data(
        self,
        gold_set_path: Path
    ) -> Tuple[List[str], List[str], List[float]]:
        """
        Create evaluation dataset from gold standard queries.

        Args:
            gold_set_path: Path to gold_set.json

        Returns:
            Tuple of (queries, code_descriptions, labels)
        """
        try:
            with open(gold_set_path, 'r') as f:
                gold_data = json.load(f)

            queries = []
            descriptions = []
            labels = []

            for item in gold_data:
                query = item.get('query', '')
                expected_codes = item.get('expected_codes', [])

                if query and expected_codes:
                    # For each expected code, create a positive pair
                    for code in expected_codes:
                        queries.append(query)
                        descriptions.append(code.get('description', ''))
                        labels.append(1.0)

            logger.info(f"Created evaluation set with {len(queries)} examples")
            return queries, descriptions, labels

        except Exception as e:
            logger.error(f"Failed to create evaluation data: {e}")
            return [], [], []

    def fine_tune(
        self,
        training_examples: List[InputExample],
        epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100
    ):
        """
        Fine-tune the model on training data.

        Args:
            training_examples: List of training examples
            epochs: Number of training epochs
            batch_size: Training batch size
            warmup_steps: Warmup steps for learning rate scheduler
        """
        logger.info(f"Starting fine-tuning for {epochs} epochs")

        # Load base model
        self.model = SentenceTransformer(self.base_model)

        # Create DataLoader
        train_dataloader = DataLoader(
            training_examples,
            shuffle=True,
            batch_size=batch_size
        )

        # Use CosineSimilarityLoss for regression on similarity scores
        train_loss = losses.CosineSimilarityLoss(self.model)

        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=str(self.output_path),
            show_progress_bar=True
        )

        logger.info(f"Fine-tuning completed. Model saved to {self.output_path}")

    def evaluate(
        self,
        eval_queries: List[str],
        eval_descriptions: List[str],
        eval_labels: List[float]
    ) -> Dict[str, float]:
        """
        Evaluate fine-tuned model.

        Args:
            eval_queries: Evaluation queries
            eval_descriptions: Code descriptions
            eval_labels: Ground truth similarity labels

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            self.model = SentenceTransformer(str(self.output_path))

        logger.info(f"Evaluating model on {len(eval_queries)} examples")

        # Create evaluator
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            sentences1=eval_queries,
            sentences2=eval_descriptions,
            scores=eval_labels,
            name='medical_coding_eval'
        )

        # Evaluate
        score = evaluator(self.model, output_path=str(self.output_path))

        logger.info(f"Evaluation score: {score:.4f}")

        return {
            "spearman_correlation": score,
            "model_path": str(self.output_path)
        }

    def compare_with_baseline(
        self,
        test_queries: List[str],
        test_codes: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare fine-tuned model with baseline.

        Args:
            test_queries: Test queries
            test_codes: Expected code descriptions

        Returns:
            Comparison metrics
        """
        logger.info("Comparing fine-tuned model with baseline")

        # Load models
        baseline_model = SentenceTransformer(self.base_model)
        finetuned_model = SentenceTransformer(str(self.output_path))

        # Encode with both models
        baseline_query_emb = baseline_model.encode(test_queries)
        baseline_code_emb = baseline_model.encode(test_codes)

        finetuned_query_emb = finetuned_model.encode(test_queries)
        finetuned_code_emb = finetuned_model.encode(test_codes)

        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity

        baseline_sims = [
            cosine_similarity([baseline_query_emb[i]], [baseline_code_emb[i]])[0][0]
            for i in range(len(test_queries))
        ]

        finetuned_sims = [
            cosine_similarity([finetuned_query_emb[i]], [finetuned_code_emb[i]])[0][0]
            for i in range(len(test_queries))
        ]

        # Calculate metrics
        baseline_avg = np.mean(baseline_sims)
        finetuned_avg = np.mean(finetuned_sims)
        improvement = ((finetuned_avg - baseline_avg) / baseline_avg) * 100

        results = {
            "baseline": {
                "avg_similarity": float(baseline_avg),
                "std_similarity": float(np.std(baseline_sims))
            },
            "finetuned": {
                "avg_similarity": float(finetuned_avg),
                "std_similarity": float(np.std(finetuned_sims))
            },
            "improvement_pct": float(improvement)
        }

        logger.info(f"Improvement: {improvement:.2f}%")

        return results

def main():
    """Demo fine-tuning workflow."""
    from data_loader import MedicalDataLoader

    # Load data
    loader = MedicalDataLoader()
    icd_codes = loader.parse_icd10_codes(Path("data/icd10_processed.csv"))

    # Initialize fine-tuner
    finetuner = MedicalEmbeddingFinetuner(
        base_model='all-MiniLM-L6-v2',
        output_path='./models/medical_embeddings'
    )

    # Create training data
    training_examples = finetuner.create_training_data_from_codes(
        icd_codes=icd_codes,
        num_samples=1000  # Use 1000 for demo, increase for production
    )

    # Fine-tune
    finetuner.fine_tune(
        training_examples=training_examples,
        epochs=2,  # Use 2 for demo, increase for production
        batch_size=16
    )

    # Evaluate
    gold_path = Path("data/gold_set.json")
    if gold_path.exists():
        eval_queries, eval_descriptions, eval_labels = finetuner.create_evaluation_data(gold_path)
        metrics = finetuner.evaluate(eval_queries, eval_descriptions, eval_labels)
        print(f"Evaluation metrics: {metrics}")

    # Compare with baseline
    test_queries = [
        "patient with acute chest pain",
        "chronic kidney disease stage 3"
    ]
    test_codes = [
        "Chest pain, unspecified",
        "Chronic kidney disease, stage 3"
    ]

    comparison = finetuner.compare_with_baseline(test_queries, test_codes)
    print(f"\nComparison with baseline:")
    print(f"  Baseline avg similarity: {comparison['baseline']['avg_similarity']:.4f}")
    print(f"  Fine-tuned avg similarity: {comparison['finetuned']['avg_similarity']:.4f}")
    print(f"  Improvement: {comparison['improvement_pct']:.2f}%")

if __name__ == "__main__":
    main()
