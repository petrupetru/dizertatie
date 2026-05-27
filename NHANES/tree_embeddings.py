import xgboost as xgb
from constants import FEATURE_NAMES, LABEL_NAME, SEED
import pandas as pd
import numpy as np


def create_tree_embeddings(df):
    """
    Creates tree-based embeddings using XGBoost Regressor.

    Args:
        df: DataFrame containing the dataset (may include NaN values for sparsity).
    Returns:
        gbr: Trained XGBoost Regressor.
        embeddings: DataFrame with tree-based embeddings.
    """
    X = df[FEATURE_NAMES].values
    y = df[LABEL_NAME].values

    y_encoded = y.astype(int)

    dtrain = xgb.DMatrix(X, label=y_encoded)

    params = {
        'n_estimators': 80,
        'max_depth': 9,
        'random_state': SEED,
        'seed': SEED,
        'objective': 'reg:squarederror',
        'verbosity': 0
    }
    gbr = xgb.train(params, dtrain, num_boost_round=80)

    dtest = xgb.DMatrix(X)
    leaf_indices = gbr.predict(dtest, pred_leaf=True)  # Shape: (n_samples, n_estimators)

    embedding_columns = [f"tree_emb_{i}" for i in range(leaf_indices.shape[1])]
    embeddings = pd.DataFrame(leaf_indices, columns=embedding_columns)

    return gbr, embeddings


def train_xgb(train_df):
    X = train_df[FEATURE_NAMES].values
    y = train_df[LABEL_NAME].values
    y_encoded = y.astype(int)

    dtrain = xgb.DMatrix(X, label=y_encoded)

    num_trees = 80

    params = {
        'n_estimators': num_trees,
        'max_depth': 9,
        'objective': 'reg:squarederror',
        'verbosity': 0,
        'seed': SEED
    }

    model = xgb.train(params, dtrain, num_boost_round=num_trees)

    return model, num_trees
