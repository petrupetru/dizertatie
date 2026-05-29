import xgboost as xgb
import numpy as np

from constants import FEATURE_NAMES, LABEL_NAME, SEED


def train_xgb(train_df):
    X = train_df[FEATURE_NAMES].values
    y = train_df[LABEL_NAME].values.astype(int)

    dtrain = xgb.DMatrix(X, label=y)

    num_trees = 80
    params = {
        "n_estimators": num_trees,
        "max_depth": 9,
        "objective": "reg:squarederror",
        "verbosity": 0,
        "seed": SEED,
    }

    model = xgb.train(params, dtrain, num_boost_round=num_trees)
    return model, num_trees
