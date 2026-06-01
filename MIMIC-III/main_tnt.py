import os
import numpy as np
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers, Model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, Dropout, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from preprocess import build_dataset
from tnt_layers import TreeEmbeddingLayer
from main import cross_validation_plot_roc, plot_confusion

def create_tnt_convnet(num_trees, max_leaves_per_tree, embedding_dim=8, l2_penalty=0.0):
    """
    Hybrid TNT-ConvNet architecture.
    Input is leaf indices for 48 hours.
    """
    total_features = num_trees * embedding_dim
    # Input shape: (48 hours, num_trees)
    leaf_input = Input(shape=(48, num_trees), name='leaf_input')

    # 1. Tree Embedding logic (Transforms indices to concatenated vectors)
    # Output shape: (batch, 48, total_features)
    tree_embeddings = TreeEmbeddingLayer(
        num_trees=num_trees, 
        max_leaves_per_tree=max_leaves_per_tree, 
        embedding_dim=embedding_dim
    )(leaf_input)

    # 2. Reshape for Conv2D: (height=48, width=total_features, channels=1)
    X = Reshape((48, total_features, 1))(tree_embeddings)

    # 3. Parallel Temporal Convolution branches (same as ISeeU)
    # Convolving over the 48 hour axis
    short_term_conv  = Conv2D(16, (3,1),  strides=(1,1), padding='same', name='3h_conv')(X)
    medium_term_conv = Conv2D(16, (6,1),  strides=(1,1), padding='same', name='6h_conv')(X)
    long_term_conv   = Conv2D(16, (12,1), strides=(1,1), padding='same', name='12h_conv')(X)

    short_term_conv  = Activation('relu')(short_term_conv)
    medium_term_conv = Activation('relu')(medium_term_conv)
    long_term_conv   = Activation('relu')(long_term_conv)

    # Pooling over the 'feature' dimension
    short_term_conv  = AveragePooling2D(pool_size=(1, 5), padding='same', name='3h_pooled')(short_term_conv)
    medium_term_conv = AveragePooling2D(pool_size=(1, 5), padding='same', name='6h_pooled')(medium_term_conv)
    long_term_conv   = AveragePooling2D(pool_size=(1, 5), padding='same', name='12h_pooled')(long_term_conv)

    X = layers.concatenate([short_term_conv, medium_term_conv, long_term_conv], axis=3)
    X = Dropout(0.45, name='dropout1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)

    X = Flatten()(X)
    X = Dense(256, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(l2_penalty))(X)
    X = Dropout(0.45, name='dropout3')(X)
    X = BatchNormalization(axis=1, name='bn3')(X)

    # Output
    output = Dense(1, activation='sigmoid', name='fc2')(X)

    return Model(inputs=leaf_input, outputs=output, name='tnt_icu_convnet')

def train_xgb_teacher(X_raw, y):
    """Trains an XGBoost model on hourly snapshots."""
    N, P, H = X_raw.shape
    # Flatten patients and hours to train GBDT on individual feature sets
    X_flat = X_raw.transpose(0, 2, 1).reshape(-1, P)
    y_flat = np.repeat(y, H)
    
    dtrain = xgb.DMatrix(X_flat, label=y_flat)
    params = {
        'max_depth': 9,
        'objective': 'binary:logistic',
        'seed': 42,
        'verbosity': 0
    }
    bst = xgb.train(params, dtrain, num_boost_round=80)
    return bst

def get_leaf_indices(bst, X_raw):
    """Extracts leaf indices for each patient and each hour."""
    N, P, H = X_raw.shape
    X_flat = X_raw.transpose(0, 2, 1).reshape(-1, P)
    ddata = xgb.DMatrix(X_flat)
    leaf_indices = bst.predict(ddata, pred_leaf=True) # (N*H, num_trees)
    return leaf_indices.reshape(N, H, -1)

def run_cv():
    input_df, labels_df = build_dataset()
    n_patients    = input_df.index.get_level_values('subject_id').nunique()
    n_predictors  = input_df.index.get_level_values('measurement_name').nunique()
    
    X_raw = input_df.values.reshape((n_patients, n_predictors, 48))
    y = labels_df['mort_icu'].values.astype(np.float32)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_idx = list(cv.split(X=np.zeros(n_patients), y=y))

    predictions, ground_truth = [], []
    os.makedirs('plots_tnt', exist_ok=True)

    for i, (train_idx, val_idx) in enumerate(cv_idx):
        print(f"\n--- Fold {i} ---")
        
        # 1. Standardize (on raw features)
        X_train_raw = X_raw[train_idx]
        X_val_raw   = X_raw[val_idx]
        
        # Simplistic mean/std per predictor
        m = np.nanmean(X_train_raw, axis=(0, 2), keepdims=True)
        s = np.nanstd(X_train_raw, axis=(0, 2), keepdims=True)
        X_tr_norm = np.nan_to_num((X_train_raw - m) / (s + 1e-8))
        X_vl_norm = np.nan_to_num((X_val_raw - m) / (s + 1e-8))

        # 2. Train XGBoost Teacher
        print("Training XGBoost teacher...")
        bst = train_xgb_teacher(X_tr_norm, y[train_idx])
        
        # 3. Generate Leaf Indices (Temporal)
        print("Generating leaf indices...")
        L_tr = get_leaf_indices(bst, X_tr_norm) # (N_tr, 48, num_trees)
        L_vl = get_leaf_indices(bst, X_vl_norm) # (N_vl, 48, num_trees)
        
        num_trees = L_tr.shape[2]
        # Extract metadata for TNT layer
        tree_df = bst.trees_to_dataframe()
        max_leaf = int(tree_df[tree_df['Feature'] == 'Leaf']['Node'].max()) + 1

        # 4. Create and Train TNT-ConvNet
        model = create_tnt_convnet(
            num_trees=num_trees, 
            max_leaves_per_tree=max_leaf
        )
        
        sgd = optimizers.SGD(learning_rate=0.01, decay=1e-7, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        ckpt = ModelCheckpoint(f'plots_tnt/fold{i}_best.keras', monitor='val_loss', save_best_only=True)
        
        model.fit(
            L_tr, y[train_idx], 
            epochs=50, 
            batch_size=32,
            class_weight={0: 1, 1: 10},
            validation_data=(L_vl, y[val_idx]),
            callbacks=[ckpt],
            verbose=1
        )

        model.load_weights(f'plots_tnt/fold{i}_best.keras')
        predictions.append(model.predict(L_vl))
        ground_truth.append(y[val_idx])

    # 5. Evaluate
    cross_validation_plot_roc(
        ground_truth, predictions, 
        'TNT-ConvNet MIMIC-III ROC', 
        filename='plots_tnt/roc_tnt.png'
    )
    plot_confusion(ground_truth, predictions, filename='plots_tnt/cm_tnt.png')

if __name__ == '__main__':
    run_cv()
