import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# ==========================================
# 1. COLLABORATIVE FILTERING (Memory Based)
# ==========================================
class CollaborativeRecommender:
    def __init__(self, train_df):
        self.train_df = train_df

        # Mappings
        self.user_ids = train_df['user_id'].unique()
        self.item_ids = train_df['item_id'].unique()

        self.user2idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item2idx = {item: i for i, item in enumerate(self.item_ids)}
        self.idx2item = {i: item for i, item in enumerate(self.item_ids)}

        # Build Matrix
        rows = train_df['user_id'].map(self.user2idx)
        cols = train_df['item_id'].map(self.item2idx)

        # Check for weights
        if 'weight' in train_df.columns:
            print("   -> Collaborative Model detected time weights. Using them.")
            data = train_df['weight'].values
        else:
            data = np.ones(len(train_df))

        self.interaction_matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        ).tocsr()

        self.item_user_matrix = self.interaction_matrix.T.tocsr()
        self.popular_items = train_df['item_id'].value_counts().head(20).index.tolist()

    def recommend(self, user_id, top_k=10, remove_seen=True):
        # Cold Start
        if user_id not in self.user2idx:
            return self.popular_items[:top_k]

        user_idx = self.user2idx[user_id]
        user_history_vec = self.interaction_matrix[user_idx]
        history_indices = user_history_vec.indices

        if len(history_indices) == 0:
            return self.popular_items[:top_k]

        # Compute Scores
        relevant_item_vecs = self.item_user_matrix[history_indices]
        user_profile = np.asarray(relevant_item_vecs.sum(axis=0)).flatten()
        scores = self.item_user_matrix.dot(user_profile)

        # Rank
        top_indices = scores.argsort()[::-1]

        recommendations = []
        for idx in top_indices:
            # Optional Filter: Remove items already seen
            if remove_seen and (idx in history_indices):
                continue

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

    def recommend(self, user_id, top_k=10, alpha=0.5, remove_seen=True):
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
        history_ids = set(user_data['item_id'].unique())

        for idx in candidate_indices:
            item_id = self.idx_to_item.get(idx)

            # Optional Filter: Remove items already seen
            if remove_seen and (item_id in history_ids):
                continue

            recommendations.append(item_id)
            if len(recommendations) >= top_k:
                break
        return recommendations


# ==========================================
# 3. HYBRID RECOMMENDER
# ==========================================
class HybridRecommender:
    def __init__(self, cf_model, content_model, content_alpha=0.4):
        self.cf_model = cf_model
        self.content_model = content_model
        self.content_alpha = content_alpha

    def recommend(self, user_id, top_k=10, hybrid_alpha=0.5, remove_seen=True):
        # 1. Get Candidates (Pass remove_seen down)
        cf_items = self.cf_model.recommend(user_id, top_k=top_k * 50, remove_seen=remove_seen)
        content_items = self.content_model.recommend(user_id, top_k=top_k * 50, alpha=self.content_alpha,
                                                     remove_seen=remove_seen)

        # 2. RRF Scoring
        cf_scores = {item: 1.0 / (i + 1) for i, item in enumerate(cf_items)}
        content_scores = {item: 1.0 / (i + 1) for i, item in enumerate(content_items)}

        # 3. Fusion
        all_items = set(cf_scores.keys()) | set(content_scores.keys())
        hybrid_scores = {}

        for item in all_items:
            s_cf = cf_scores.get(item, 0.0)
            s_content = content_scores.get(item, 0.0)
            hybrid_scores[item] = (hybrid_alpha * s_cf) + ((1 - hybrid_alpha) * s_content)

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