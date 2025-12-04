import os

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from src.models import CollaborativeRecommender, ContentRecommender, HybridRecommender

# ==========================================
# 1. VISUALIZATION HELPER
# ==========================================
def get_book_titles(item_ids, full_catalog):
    """
    Maps a list of Item IDs to their metadata (Title, Author).
    Args:
        item_ids: List of item IDs.
        full_catalog: The DataFrame containing 'item_id', 'Title', 'Author', 'Subjects' columns.'.
    """
    # Filter catalog for these items
    subset = full_catalog[full_catalog['item_id'].isin(item_ids)]

    # Drop duplicates to ensure unique book info per ID
    subset = subset[['item_id', 'Title', 'Author', 'Subjects']].drop_duplicates(subset='item_id')

    # Re-index to match the input order (Rank 1, Rank 2...)
    return subset.set_index('item_id').reindex(item_ids)


# ==========================================
# 2. GENERIC ALPHA TUNER (Content & Hybrid)
# ==========================================
def tune_alpha(model, test_df, param_name='alpha', k=10, values=[0.0, 0.2, 0.5, 0.8, 1.0]):
    """
    Tunes a float parameter (like alpha or hybrid_alpha) based on Hit Rate @ k.

    Args:
        model: The instantiated model object (ContentRecommender or HybridRecommender).
        test_df: Validation DataFrame (user_id, item_id).
        param_name: Name of the argument to pass to .recommend() (e.g. 'alpha').
        k: Top-K items to evaluate.
        values: List of float values to test.

    Returns:
        best_value: The winning parameter value.
    """
    print(f">>> Tuning '{param_name}' on {len(test_df)} users...")

    results = {}

    for val in values:
        hits = 0
        total = 0

        # Determine kwargs dynamically
        kwargs = {param_name: val}

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{param_name}={val}"):
            user_id = row['user_id']
            target = row['item_id']

            try:
                # Call recommend with dynamic parameter
                # e.g., model.recommend(user_id, top_k=10, alpha=0.2)
                recs = model.recommend(user_id, top_k=k, **kwargs)

                if target in recs:
                    hits += 1
            except Exception:
                pass
            total += 1

        score = hits / total if total > 0 else 0
        results[val] = score
        print(f"   [{val:.1f}] Hit Rate: {score:.8%}")

    best_val = max(results, key=results.get)
    print(f"\n>>> Best {param_name}: {best_val} (Hit Rate: {results[best_val]:.8%})")

    return best_val


# ==========================================
# AGILE TUNING: SAMPLE BY USER
# ==========================================
def get_tuning_sample(test_df, n_users=1000, random_state=42):
    """
    Creates a smaller test set by picking N random users
    and keeping ALL their interactions (preserving time structure).
    """
    # 1. Get all unique users in Test
    unique_users = test_df['user_id'].unique()

    # 2. Pick N random users
    if len(unique_users) > n_users:
        np.random.seed(random_state)
        sampled_users = np.random.choice(unique_users, size=n_users, replace=False)

        # 3. Filter the Test DF to only these users
        tuning_df = test_df[test_df['user_id'].isin(sampled_users)].copy()

        print(f"   -> Sampling Strategy: Selected {n_users} Users")
        print(f"   -> Original Rows: {len(test_df)} | Sampled Rows: {len(tuning_df)}")
        return tuning_df
    else:
        # If we have fewer users than requested, just return the whole set
        return test_df


# ==========================================
# 4. KAGGLE SUBMISSION GENERATOR
# ==========================================
def generate_kaggle_submission(model, target_user_ids, k=10, remove_seen=True, **kwargs):
    """
    Generates a submission DataFrame for ANY model class.

    Args:
        model: The trained model object (must have .recommend() method).
        target_user_ids (list): List of user_ids to predict for.
        k (int): Number of items to recommend per user.
        **kwargs: Extra arguments for .recommend (e.g., hybrid_alpha=0.6, alpha=0.2).

    Returns:
        pd.DataFrame: DataFrame with columns ['user_id', 'recommendation'] ready for CSV export.
    """
    print(f">>> Generating predictions for {len(target_user_ids)} users...")

    recommendations = []

    # Progress bar to track generation
    # Using tqdm.auto to work in both notebooks and scripts
    for user_id in tqdm(target_user_ids, desc="Generating Submission"):
        try:
            # Polymorphic call: Works for CF, Content, Hybrid, or SVD
            # We explicitly pass remove_seen, and unpack any other tuning params (alphas)
            recs = model.recommend(user_id, top_k=k, **kwargs)

            # Format: Space-separated string "id1 id2 id3..."
            recs_str = " ".join([str(r) for r in recs])
        except Exception as e:
            # Fallback for errors (safeguard)
            print(f"Error for user {user_id}: {e}")
            raise e

        recommendations.append(recs_str)

    # Create DataFrame matching Kaggle format
    submission = pd.DataFrame({
        'user_id': target_user_ids,
        'recommendation': recommendations
    })

    # Sort by user_id to ensure consistent ordering
    submission.sort_values(by='user_id', inplace=True)

    return submission


# ==========================================
# 8. DATA TUNING: HALF-LIFE OPTIMIZATION
# ==========================================
def tune_half_life(loader, test_df, item_tfidf, item_minilm, item_map,
                   evaluator, best_c_alpha, best_h_alpha, k=10):
    checkpoint_path = '../data/artifacts/half_life_results.pkl'

    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(">>> Tuning Half-Life (Data Decay) with Checkpointing...")

    # Load existing results if they exist
    if os.path.exists(checkpoint_path):
        results = joblib.load(checkpoint_path)
        results = {int(k): v for k, v in results.items()}
        print(f"   -> Found existing checkpoint with {len(results)} results.")
    else:
        results = {}

    # Values to test (in days)
    half_lives = [30, 60, 90, 120, 180, 365]

    for hl in half_lives:
        if int(hl) in results:
            print(f"\n--- [Skipping] Half-Life = {hl} days (Found: {results[hl]:.4%}) ---")
            continue

        print(f"\n--- Testing Half-Life = {hl} days ---")

        try:
            # 1. Regenerate Training Data with new weights
            # We get the new split/weights
            train_weighted, _ = loader.get_time_split(train_ratio=0.8, half_life_days=hl)

            # 2. Re-Train Collaborative Model
            # It will automatically pick up the new 'weight' column
            cf_new = CollaborativeRecommender(train_weighted)

            # 3. Re-Train Content Model
            # We reuse the artifacts (tfidf/vectors) we loaded earlier
            content_new = ContentRecommender(
                train_weighted,
                item_tfidf,
                item_minilm,
                item_map
            )

            # 4. Re-Build Hybrid
            # We use the best alphas we found earlier
            hybrid_new = HybridRecommender(
                cf_new,
                content_new,
                content_alpha=best_c_alpha
            )

            # 5. Evaluate
            # We evaluate the Hybrid using the best known hybrid_alpha
            metrics = evaluator.evaluate(
                hybrid_new,
                test_df,
                k=k,
                model_name=f"Hybrid (HL={hl})",
                hybrid_alpha=best_h_alpha
            )

            score = metrics['Hit Rate @ 10']
            print(f">>> Hit Rate: {score:.4%} at HL={hl} days")
            results[hl] = score
            joblib.dump(results, checkpoint_path)
            print(f"   -> Saved Checkpoint: {checkpoint_path}")

        except Exception as e:
            print(f"   >>> Error with HL={hl}: {e}")

    # Summary
    print("\n>>> Evaluation Summary:")
    for hl, score in results.items():
        print(f"   -> Hit Rate: {score:.4%} at HL={hl} days")
    # Find Winner
    best_hl = max(results, key=results.get)
    print(f"\n>>> Best Half-Life: {best_hl} days")
    print(f"   -> Hit Rate: {results[best_hl]:.4%}")

    return best_hl