import os
import numpy as np
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers, Model
from tensorflow.keras.layers import (Input, Activation, BatchNormalization, Flatten, 
                                     Conv2D, AveragePooling2D, Dropout, Reshape, Dense)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from preprocess import build_dataset
from tnt_layers import TreeEmbeddingLayer
from main import cross_validation_plot_roc, plot_confusion

def create_tnt_convnet(num_trees, max_leaves_per_tree, embedding_dim=4, l2_penalty=0.1):
    """
    Hybrid TNT-ConvNet architecture.
    Input: Leaf indices for 48 hours.
    Neural Component: ISeeU ConvNet architecture.
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

    # 3. Parallel Temporal Convolution branches (Short, Medium, Long term)
    short_term_conv  = Conv2D(16, (3,1),  strides=(1,1), padding='same', name='3h_conv')(X)
    medium_term_conv = Conv2D(16, (6,1),  strides=(1,1), padding='same', name='6h_conv')(X)
    long_term_conv   = Conv2D(16, (12,1), strides=(1,1), padding='same', name='12h_conv')(X)

    short_term_conv  = Activation('relu')(short_term_conv)
    medium_term_conv = Activation('relu')(medium_term_conv)
    long_term_conv   = Activation('relu')(long_term_conv)

    # Average pooling across the feature width
    short_term_conv  = AveragePooling2D(pool_size=(1, 5), padding='same', name='3h_pooled')(short_term_conv)
    medium_term_conv = AveragePooling2D(pool_size=(1, 5), padding='same', name='6h_pooled')(medium_term_conv)
    long_term_conv   = AveragePooling2D(pool_size=(1, 5), padding='same', name='12h_pooled')(long_term_conv)

    X = layers.concatenate([short_term_conv, medium_term_conv, long_term_conv], axis=3)
    X = Dropout(0.45, name='dropout1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)

    # 4. Dense Head
    X = Flatten()(X)
    X = Dense(128, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(l2_penalty))(X)
    X = Dropout(0.45, name='dropout3')(X)
    X = BatchNormalization(axis=1, name='bn3')(X)

    # Final Binary Classification
    output = Dense(1, activation='sigmoid', name='fc2', kernel_regularizer=regularizers.l2(l2_penalty))(X)

    return Model(inputs=leaf_input, outputs=output, name='tnt_icu_convnet')

def train_xgb_teacher(X_raw, y):
    """Trains an XGBoost teacher on clinical snapshots (one row per patient-hour)."""
    N, P, H = X_raw.shape
    # Reshape to (Samples * Hours, Features) for XGBoost training
    X_flat = X_raw.transpose(0, 2, 1).reshape(-1, P)
    y_flat = np.repeat(y, H)
    
    dtrain = xgb.DMatrix(X_flat, label=y_flat)
    params = {
        'max_depth': 3,
        'objective': 'binary:logistic',
        'seed': 42,
        'verbosity': 0
    }
    bst = xgb.train(params, dtrain, num_boost_round=60)
    return bst

def get_leaf_indices(bst, X_raw):
    """Passes raw features through XGBoost to get temporal leaf indices."""
    N, P, H = X_raw.shape
    X_flat = X_raw.transpose(0, 2, 1).reshape(-1, P)
    ddata = xgb.DMatrix(X_flat)
    leaf_indices = bst.predict(ddata, pred_leaf=True) 
    return leaf_indices.reshape(N, H, -1)

def run_cv():
    input_df, labels_df = build_dataset(force=False, sparse=True)
    n_patients    = input_df.index.get_level_values('subject_id').nunique()
    n_predictors  = input_df.index.get_level_values('measurement_name').nunique()
    
    X_raw = input_df.values.reshape((n_patients, n_predictors, 48))
    y = labels_df['mort_icu'].values.astype(np.float32)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_idx = list(cv.split(X=np.zeros(n_patients), y=y))

    predictions, ground_truth, fold_aurocs, fold_auprcs = [], [], [], []
    os.makedirs('models/tnt', exist_ok=True)

    for i, (train_idx, val_idx) in enumerate(cv_idx):
        print(f"\n--- Fold {i} ---")
        
        # Normalize features
        X_train_raw = X_raw[train_idx]
        X_val_raw   = X_raw[val_idx]
        m = np.nanmean(X_train_raw, axis=(0, 2), keepdims=True)
        s = np.nanstd(X_train_raw, axis=(0, 2), keepdims=True)

        # Save normalization parameters for later inference/distillation
        np.save(f'models/tnt/fold{i}_mean.npy', m)
        np.save(f'models/tnt/fold{i}_std.npy', s)

        X_tr_norm = np.nan_to_num((X_train_raw - m) / (s + 1e-8))
        X_vl_norm = np.nan_to_num((X_val_raw - m) / (s + 1e-8))

        # 1. XGBoost Feature Engineering
        print("Training XGBoost teacher...")
        bst = train_xgb_teacher(X_tr_norm, y[train_idx])
        bst.save_model(f'models/tnt/fold{i}_xgb_teacher.json')
        
        # 2. Extract Leaf Sequences
        print("Generating leaf indices...")
        L_tr = get_leaf_indices(bst, X_tr_norm) 
        L_vl = get_leaf_indices(bst, X_vl_norm) 
        
        num_trees = L_tr.shape[2]
        tree_df = bst.trees_to_dataframe()
        max_leaf = int(tree_df['Node'].max()) + 1

        # 3. Model Training
        model = create_tnt_convnet(
            num_trees=num_trees, 
            max_leaves_per_tree=max_leaf,
            embedding_dim=4,
            l2_penalty=0.01
        )
        
        sgd = optimizers.SGD(learning_rate=0.01, decay=1e-7, momentum=0.9, nesterov=True)
        model.compile(
            optimizer=sgd, 
            loss='binary_crossentropy', 
            metrics=['accuracy', 
                     tf.keras.metrics.AUC(name='auroc'), 
                     tf.keras.metrics.AUC(name='auprc', curve='PR')]
        )

        ckpt_path = f'models/tnt/fold{i}_best.keras'
        ckpt = ModelCheckpoint(ckpt_path, monitor='val_auprc', save_best_only=True, mode='max')
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_auprc', factor=0.5, patience=5, mode='max', verbose=1)
        
        model.fit(
            L_tr, y[train_idx], 
            epochs=100, 
            batch_size=256,
            class_weight={0: 1, 1: 10},
            validation_data=(L_vl, y[val_idx]),
            callbacks=[ckpt, early_stop, reduce_lr],
            verbose=1
        )

        model.load_weights(ckpt_path)
        fold_preds = model.predict(L_vl)
        predictions.append(fold_preds)
        ground_truth.append(y[val_idx])

        fold_auc = roc_auc_score(y[val_idx], fold_preds)
        fold_prc = average_precision_score(y[val_idx], fold_preds)
        print(f"Fold {i} Saved Model -> AUROC: {fold_auc:.4f}, AUPRC: {fold_prc:.4f}")
        fold_aurocs.append(fold_auc)
        fold_auprcs.append(fold_prc)

    print(f"\nFinal TNT-ConvNet {len(cv_idx)}-fold CV Results:")
    print(f"Mean AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
    print(f"Mean AUPRC: {np.mean(fold_auprcs):.4f} ± {np.std(fold_auprcs):.4f}")

    # 4. Global Evaluation
    cross_validation_plot_roc(
        ground_truth, predictions, 
        'TNT-ConvNet MIMIC-III Mortality Prediction ROC', 
        filename='models/tnt/roc_tnt.png'
    )
    plot_confusion(ground_truth, predictions, filename='models/tnt/cm_tnt.png')

if __name__ == '__main__':
    run_cv()