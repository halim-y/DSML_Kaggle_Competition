from tqdm import tqdm
import time
from joblib import Parallel, delayed

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
        print(f"   [{val:.1f}] Hit Rate: {score:.4%}")

    best_val = max(results, key=results.get)
    print(f"\n>>> Best {param_name}: {best_val} (Hit Rate: {results[best_val]:.4%})")

    return best_val


# ==========================================
# 3. SVD TUNER (Re-Training Loop)
# ==========================================
def tune_svd(train_df, test_df, k=10, components=[10, 25, 50, 100, 200, 300]):
    """
    Tunes n_components for SVD. Requires re-training the model each time.

    Args:
        train_df: Training data for fitting SVD.
        test_df: Validation data for evaluating Hit Rate.
        k: Top-K items.
        components: List of component counts to test.

    Returns:
        best_n: The optimal number of components.
    """
    # Import inside function to avoid circular dependency issues at top level
    from src.models import SVDRecommender

    print(f">>> Tuning SVD n_components...")

    results = {}

    for n in components:
        print(f"\n--- Training SVD with n={n} ---")
        start_time = time.time()

        try:
            # 1. Train New Model
            model = SVDRecommender(train_df, n_components=n)

            # 2. Evaluate Hit Rate
            hits = 0
            total = 0

            for _, row in test_df.iterrows():
                user_id = row['user_id']
                target = row['item_id']

                # SVD Recommend
                recs = model.recommend(user_id, top_k=k)
                if target in recs:
                    hits += 1
                total += 1

            score = hits / total if total > 0 else 0
            results[n] = score

            elapsed = time.time() - start_time
            print(f"   -> Hit Rate: {score:.4%} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"   -> Failed for n={n}: {e}")
            results[n] = 0.0

    best_n = max(results, key=results.get)
    print(f"\n>>> Best SVD n_components: {best_n} (Hit Rate: {results[best_n]:.4%})")

    return best_n