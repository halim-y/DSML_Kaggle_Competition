import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class ModelEvaluator:
    def __init__(self, train_df, full_catalog_ids):
        """
        Comprehensive Evaluator for Recommender Systems.

        Args:
            train_df: The training DataFrame (used to calculate Item Popularity for Novelty).
            full_catalog_ids: List/Set of all unique item IDs (used to calculate Coverage).
        """
        self.train_df = train_df
        self.catalog_items = set(full_catalog_ids)

        # Pre-compute Item Popularity (Probability) for Novelty metrics
        # P(i) = count(i) / total_interactions
        print("   -> Pre-computing Item Popularity for Novelty metrics...")
        item_counts = train_df['item_id'].value_counts()
        total_interactions = len(train_df)
        self.item_probs = (item_counts / total_interactions).to_dict()

    def _calculate_novelty(self, recs):
        """
        Calculates mean self-information of a list of recommendations.
        Formula: -log2(P(i))
        """
        novelty_score = 0
        for item in recs:
            # P(i) gets a small epsilon if missing to avoid log(0) for truly cold items
            p = self.item_probs.get(item, 1e-10)
            novelty_score += -np.log2(p)

        return novelty_score / len(recs) if len(recs) > 0 else 0

    def evaluate(self, model, test_df, k=10, model_name="Model", **kwargs):
        """
        Runs the full battery of metrics on the test set.

        Args:
            model: The recommender object (must have .recommend() method).
            test_df: Validation DataFrame.
            k: Number of recommendations to generate.
            model_name: Label for the report.
            **kwargs: Extra parameters passed to model.recommend() (e.g., alpha, hybrid_alpha).
        """
        print(f">>> Evaluating {model_name} on {len(test_df)} users...")

        metrics = {
            'hits': 0,  # Hit Rate
            'sum_ap': 0.0,  # MAP (Reciprocal Rank)
            'sum_nov': 0.0,  # Novelty
            'total': 0
        }

        # Track unique recommended items for Coverage
        unique_recs = set()

        # Using tqdm.auto for environment-agnostic progress bars
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Eval {model_name}"):
            user_id = row['user_id']
            target = row['item_id']

            try:
                # Pass **kwargs dynamically (alphas, etc.)
                recs = model.recommend(user_id, top_k=k, **kwargs)

                # 1. Update Coverage Set
                unique_recs.update(recs)

                # 2. Hit Rate & MAP (Reciprocal Rank for LOO)
                if target in recs:
                    metrics['hits'] += 1
                    rank = recs.index(target) + 1
                    metrics['sum_ap'] += (1.0 / rank)

                # 3. Novelty
                metrics['sum_nov'] += self._calculate_novelty(recs)

            except Exception:
                # Fail gracefully if model errors out
                pass

            metrics['total'] += 1

        # Compile Final Scores
        n = metrics['total'] if metrics['total'] > 0 else 1

        final_scores = {
            'Model': model_name,
            'Hit Rate @ 10': metrics['hits'] / n,
            'MAP @ 10': metrics['sum_ap'] / n,
            'Novelty': metrics['sum_nov'] / n,
            'Coverage': len(unique_recs) / len(self.catalog_items)
        }

        return final_scores

    def evaluate_segments(self, model, test_df, k=10, model_name="Model", **kwargs):
        """
        Analyzes performance on 'Warm' vs 'Cold' items separately.
        Useful to prove why Content/Hybrid models are necessary.
        """
        print(f">>> Segmenting Evaluation for {model_name}...")

        # 1. Identify Warm vs Cold Items
        train_items = set(self.train_df['item_id'].unique())

        # Boolean mask: Is the target item inside the training set?
        is_warm = test_df['item_id'].isin(train_items)

        test_warm = test_df[is_warm]
        test_cold = test_df[~is_warm]

        print(f"   -> Warm Test Cases: {len(test_warm)}")
        print(f"   -> Cold Test Cases: {len(test_cold)}")

        # 2. Helper for simple Hit Rate
        def get_hit_rate(sub_df):
            hits = 0
            total = 0
            for _, row in sub_df.iterrows():
                try:
                    recs = model.recommend(row['user_id'], top_k=k, **kwargs)
                    if row['item_id'] in recs:
                        hits += 1
                except:
                    pass
                total += 1
            return hits / total if total > 0 else 0

        # 3. Calculate
        hr_warm = get_hit_rate(test_warm)
        hr_cold = get_hit_rate(test_cold)

        return {
            'Model': model_name,
            'HR@10 (Warm)': hr_warm,
            'HR@10 (Cold)': hr_cold
        }