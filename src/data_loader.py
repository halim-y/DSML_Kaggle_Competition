import pandas as pd
import numpy as np
import scipy.sparse as sp


class DataLoader:
    def __init__(self, base_path='../data'):
        """
        Data Loader.
        base_path: The folder containing 'raw', 'processed', and 'artifacts'.
        """
        self.base_path = base_path

        # Internal storage
        self.full_df = None
        self.train_df = None
        self.test_df = None
        self.tfidf = None
        self.embeddings = None
        self.item_map = {}

    def load_data(self):
        """Loads the main interactions dataset."""
        print(">>> Loading Interactions...")

        # Load the merged interactions (User History)
        path = f"{self.base_path}/processed/interaction_items_merged.csv"
        self.full_df = pd.read_csv(path)

        # Ensure timestamp is numeric
        self.full_df['timestamp'] = pd.to_numeric(self.full_df['timestamp'], errors='coerce')

        print(f"   -> Interactions loaded: {len(self.full_df)} rows")
        return self

    def _process_ranking_and_weight(self, df, half_life_days):
        """
        Helper: Sorts, calculates pct_rank, and calculates time decay weights.
        """
        # 1. Sort strictly by User and Time
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # 2. Calculate Percentage Rank
        # 0.0 = First interaction, 1.0 = Last interaction
        df['pct_rank'] = df.groupby('user_id')['timestamp'].rank(method='first', pct=True)

        # 3. Add Time Decay Weight
        # We calculate age relative to the LATEST timestamp in this specific dataframe
        max_time = df['timestamp'].max()
        days_ago = (max_time - df['timestamp']) / 86400

        # Formula: 0.5 ^ (days / half_life)
        # Recent = 1.0, Old = 0.0
        df['weight'] = np.power(0.5, days_ago / half_life_days)

        return df

    def get_time_split(self, train_ratio=0.8, half_life_days=120):
        """
        Returns Train/Test based on pct_rank.
        Applies weights ONLY to Train (to simulate historical training).
        """
        if self.full_df is None:
            self.load_data()

        print(f">>> Splitting Data (First {train_ratio * 100:.0f}% Train)...")

        # 1. Calculate Ranks on the full set first to determine the split line
        df = self.full_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        df['pct_rank'] = df.groupby('user_id')['timestamp'].rank(method='first', pct=True)

        # 2. Split
        self.train_df = df[df['pct_rank'] <= train_ratio].copy()
        self.test_df = df[df['pct_rank'] > train_ratio].copy()

        # 3. Apply Time Decay to TRAIN set
        # Note: We recalculate weights specifically for the training context
        print(f"   -> Applying Time Decay (Half-Life: {half_life_days} days)...")

        max_time_train = self.train_df['timestamp'].max()
        days_ago = (max_time_train - self.train_df['timestamp']) / 86400
        self.train_df['weight'] = np.power(0.5, days_ago / half_life_days)

        print(f"   -> Train: {len(self.train_df)} | Test: {len(self.test_df)}")
        print(f"   -> Total: {len(self.train_df)} + {len(self.test_df)} = {len(self.full_df)}")
        return self.train_df, self.test_df

    def get_full_data(self, half_life_days=120):
        """
        Returns the FULL dataset sorted, ranked, and weighted.
        Used for Final Production Training.
        """
        if self.full_df is None:
            self.load_data()

        print(f">>> Preparing Full Dataset (Half-Life: {half_life_days} days)...")

        # Use the helper to process the whole thing
        processed_df = self._process_ranking_and_weight(self.full_df.copy(), half_life_days)

        print(f"   -> Full Data Weighted: {len(processed_df)} rows")
        return processed_df

    def get_content_data(self):
        """
        Loads matrices and builds the map directly from items_cleaned.csv
        """
        print(">>> Loading Content Artifacts...")

        # 1. Load Matrices
        self.tfidf = sp.load_npz(f"{self.base_path}/artifacts/tfidf_matrix.npz")
        self.embeddings = np.load(f"{self.base_path}/artifacts/minilm_embeddings.npy")

        # 2. Load Items to build the Map
        items_path = f"{self.base_path}/processed/items_cleaned.csv"
        items_df = pd.read_csv(items_path)

        # Map Item_ID -> Row Index
        self.item_map = {item_id: idx for idx, item_id in enumerate(items_df['item_id'].values)}

        print(f"   -> Loaded features for {len(self.item_map)} items.")
        return self.tfidf, self.embeddings, self.item_map