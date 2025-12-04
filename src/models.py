import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# ==========================================
# 1. COLLABORATIVE FILTERING (Memory Based)
# ==========================================
class CollaborativeRecommender:
    def __init__(self, train_df):
        """
        Item-Item Collaborative Filtering using TF-IDF weighted interactions.
        This down-weights popular items to find more meaningful similarities.
        """
        self.train_df = train_df

        # Mappings
        self.user_ids = train_df['user_id'].unique()
        self.item_ids = train_df['item_id'].unique()

        self.user2idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item2idx = {item: i for i, item in enumerate(self.item_ids)}
        self.idx2item = {i: item for i, item in enumerate(self.item_ids)}

        # Build Matrix (Rows=Users, Cols=Items)
        rows = train_df['user_id'].map(self.user2idx)
        cols = train_df['item_id'].map(self.item2idx)

        # Use Time Decay weights if available, else 1.0
        if 'weight' in train_df.columns:
            data = train_df['weight'].values
        else:
            data = np.ones(len(train_df))

        # 1. Create Raw Interaction Matrix
        self.raw_matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        ).tocsr()

        # 2. Apply TF-IDF Transformation
        # TF = Time Decay Weight (Strength of interaction)
        # IDF = log(Total Users / Users who read Item i) -> Penalizes popular items
        print("   -> Applying TF-IDF to Interaction Matrix...")
        self.tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
        self.weighted_matrix = self.tfidf_transformer.fit_transform(self.raw_matrix)

        # 3. Transpose for Item-Item Calculations (Items x Users)
        # We work with the weighted matrix now, not the raw one
        self.item_user_matrix = self.weighted_matrix.T.tocsr()

        # Pre-compute Global Popularity (Fallback)
        self.popular_items = train_df['item_id'].value_counts().head(20).index.tolist()

    def recommend(self, user_id, top_k=10):
        # Cold Start
        if user_id not in self.user2idx:
            return self.popular_items[:top_k]

        user_idx = self.user2idx[user_id]

        # Get raw history indices (to filter 'seen' items later)
        raw_history_vec = self.raw_matrix[user_idx]
        history_indices = raw_history_vec.indices

        if len(history_indices) == 0:
            return self.popular_items[:top_k]

        # Compute Scores using the Weighted Matrix
        # 1. Get vectors for items the user liked (from the weighted matrix)
        relevant_item_vecs = self.item_user_matrix[history_indices]

        # 2. Build User Profile (Sum of item vectors)
        # This profile now represents "Weighted Taste" rather than just "Books Read"
        user_profile = np.asarray(relevant_item_vecs.sum(axis=0)).flatten()

        # 3. Compute Similarity (Dot Product)
        # Items that are similar to the User's unique taste profile will score higher
        scores = self.item_user_matrix.dot(user_profile)

        # Rank
        top_indices = scores.argsort()[::-1]

        recommendations = []
        for idx in top_indices:
            recommendations.append(self.idx2item[idx])

            if len(recommendations) >= top_k:
                break
        return recommendations

# ==========================================
# 2. CONTENT-BASED FILTERING
# ==========================================
class ContentRecommender:
    def __init__(self, interactions_df, tfidf_matrix, minilm_embeddings, item_to_row_idx):
        self.interactions_df = interactions_df
        self.tfidf_matrix = tfidf_matrix
        self.minilm_embeddings = minilm_embeddings
        self.item_map = item_to_row_idx
        self.idx_to_item = {v: k for k, v in item_to_row_idx.items()}
        self.popular_items = interactions_df['item_id'].value_counts().head(20).index.tolist()

    def recommend(self, user_id, top_k=10, alpha=0.5):
        # 1. Get History
        user_data = self.interactions_df[self.interactions_df['user_id'] == user_id]

        if len(user_data) == 0:
            return self.popular_items[:top_k]

        # 2. Get Indices & Weights
        valid_indices = []
        weights = []
        has_weights = 'weight' in user_data.columns

        for _, row in user_data.iterrows():
            if row['item_id'] in self.item_map:
                valid_indices.append(self.item_map[row['item_id']])
                weights.append(row['weight'] if has_weights else 1.0)

        if not valid_indices:
            return self.popular_items[:top_k]

        # 3. Build Profile
        user_prof_mini = np.average(self.minilm_embeddings[valid_indices], axis=0, weights=weights).reshape(1, -1)
        user_prof_tfidf = np.asarray(self.tfidf_matrix[valid_indices].mean(axis=0))

        # 4. Compute Similarity
        sim_mini = cosine_similarity(user_prof_mini, self.minilm_embeddings).flatten()
        sim_tfidf = cosine_similarity(user_prof_tfidf, self.tfidf_matrix).flatten()

        final_scores = (alpha * sim_mini) + ((1 - alpha) * sim_tfidf)

        # 5. Rank
        candidate_indices = final_scores.argsort()[::-1]

        recommendations = []

        for idx in candidate_indices:
            item_id = self.idx_to_item.get(idx)
            recommendations.append(item_id)
            if len(recommendations) >= top_k:
                break
        return recommendations


# ==========================================
# 3. HYBRID RECOMMENDER
# ==========================================
class HybridRecommender:
    def __init__(self, cf_model, content_model, content_alpha=0.5):
        self.cf_model = cf_model
        self.content_model = content_model
        self.content_alpha = content_alpha

        # Pre-compute Global Popularity Scores (Normalized 0-1)
        # We use the CF model's training data for this
        item_counts = cf_model.train_df['item_id'].value_counts()
        max_count = item_counts.max()
        self.pop_scores = (item_counts / max_count).to_dict()

    def recommend(self, user_id, top_k=10, hybrid_alpha=0.5, pop_weight=0.1):
        """
        Hybrid Recommendation with Popularity Injection.

        pop_weight: Weight for Global Popularity (0.0 to 1.0).
                    Small values (e.g., 0.1) help stabilize predictions.
        """
        # 1. Get Candidates
        # Fetch more candidates to allow re-ranking
        cf_items = self.cf_model.recommend(user_id, top_k=top_k * 50)
        content_items = self.content_model.recommend(user_id, top_k=top_k * 50, alpha=self.content_alpha)

        # 2. RRF Scoring (Rank 0 -> 1.0)
        cf_scores = {item: 1.0 / (i + 1) for i, item in enumerate(cf_items)}
        content_scores = {item: 1.0 / (i + 1) for i, item in enumerate(content_items)}

        # 3. Fusion
        all_items = set(cf_scores.keys()) | set(content_scores.keys())
        hybrid_scores = {}

        # Normalize weights to sum to 1.0 (approximately)
        # We treat pop_weight as an additive boost or a third component
        # Strategy: (1 - pop_weight) * [Hybrid] + pop_weight * [Pop]

        for item in all_items:
            s_cf = cf_scores.get(item, 0.0)
            s_content = content_scores.get(item, 0.0)
            s_pop = self.pop_scores.get(item, 0.0)

            # Base Hybrid Score
            base_score = (hybrid_alpha * s_cf) + ((1 - hybrid_alpha) * s_content)

            # Inject Popularity
            final_score = base_score + (pop_weight * s_pop)

            hybrid_scores[item] = final_score

        sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_items[:top_k]]


# ==========================================
# 4. SVD RECOMMENDER
# ==========================================
class SVDRecommender:
    def __init__(self, train_df, n_components=50):
        self.train_df = train_df
        self.user_ids = train_df['user_id'].unique()
        self.item_ids = train_df['item_id'].unique()
        self.user2idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item2idx = {item: i for i, item in enumerate(self.item_ids)}
        self.idx2item = {i: item for i, item in enumerate(self.item_ids)}

        rows = train_df['user_id'].map(self.user2idx)
        cols = train_df['item_id'].map(self.item2idx)

        # Check for weights
        if 'weight' in train_df.columns:
            data = train_df['weight'].values
        else:
            data = np.ones(len(train_df))

        self.interaction_matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        ).tocsr()

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd.fit_transform(self.interaction_matrix)
        self.item_factors = self.svd.components_.T
        self.popular_items = train_df['item_id'].value_counts().head(20).index.tolist()

    def recommend(self, user_id, top_k=10, remove_seen=True):
        if user_id not in self.user2idx:
            return self.popular_items[:top_k]

        user_idx = self.user2idx[user_id]
        user_vec = self.user_factors[user_idx].reshape(1, -1)
        scores = user_vec.dot(self.item_factors.T).flatten()

        # Optional Filter: Mask items already seen
        if remove_seen:
            history_indices = self.interaction_matrix[user_idx].indices
            scores[history_indices] = -np.inf

        top_indices = scores.argsort()[-(top_k):][::-1]
        return [self.idx2item[i] for i in top_indices]