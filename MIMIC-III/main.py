"""
ISeeU ConvNet for ICU mortality prediction on MIMIC-III.
Model architecture copied from the ISeeU paper (model-3 / create_model3).
"""
import os
import numpy as np
np.random.seed(1)

import itertools
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.set_random_seed(1)

from tensorflow.keras import layers, regularizers, optimizers
from tensorflow.keras.layers import (Input, Dense, Activation, BatchNormalization,
                                     Flatten, Conv2D, AveragePooling2D, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix

from preprocess import build_dataset

plt.style.use('ggplot')


# ── ISeeU model-3 (the model used in the paper) ───────────────────────────────

def create_model(input_shape, use_dropout=True, conv_drop_prob=0.45, fc_drop_prob=0.45, l2_penalty=0.0):
    X_input = Input(input_shape, name='input')

    # First layer – three parallel conv branches
    short_term_conv  = Conv2D(16, (3,1),  strides=(1,1), padding='same', name='3h_conv')(X_input)
    medium_term_conv = Conv2D(16, (6,1),  strides=(1,1), padding='same', name='6h_conv')(X_input)
    long_term_conv   = Conv2D(16, (12,1), strides=(1,1), padding='same', name='12h_conv')(X_input)

    short_term_conv  = Activation('relu')(short_term_conv)
    medium_term_conv = Activation('relu')(medium_term_conv)
    long_term_conv   = Activation('relu')(long_term_conv)

    short_term_conv  = AveragePooling2D(pool_size=(1,3), padding='same', name='3h_pooled')(short_term_conv)
    medium_term_conv = AveragePooling2D(pool_size=(1,3), padding='same', name='6h_pooled')(medium_term_conv)
    long_term_conv   = AveragePooling2D(pool_size=(1,3), padding='same', name='12h_pooled')(long_term_conv)

    X = layers.concatenate([short_term_conv, medium_term_conv, long_term_conv], axis=3)

    if use_dropout:
        X = Dropout(conv_drop_prob, name='dropout1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)

    # Fully connected
    X = Flatten()(X)
    X = Dense(256, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(l2_penalty))(X)
    if use_dropout:
        X = Dropout(fc_drop_prob, name='dropout3')(X)
    X = BatchNormalization(axis=1, name='bn3')(X)

    # Output
    X = Dense(1, activation='sigmoid', name='fc2', kernel_regularizer=regularizers.l2(l2_penalty))(X)

    return Model(inputs=X_input, outputs=X, name='icu_mortality_conv')


# ── plotting helpers ──────────────────────────────────────────────────────────

def cross_validation_plot_roc(ground_truth_list, predictions_list, title, filename=None):
    plt.figure(figsize=(12, 8))
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    for i, (g, p) in enumerate(zip(ground_truth_list, predictions_list)):
        fpr, tpr, _ = roc_curve(y_true=g, y_score=p, drop_intermediate=False)
        tprs.append(np.interp(mean_fpr, fpr, tpr)); tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr); aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i} (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1], '--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr); std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=.8,
             label=rf'Mean ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})')
    std_tpr = np.std(tprs, axis=0)
    plt.fill_between(mean_fpr, np.maximum(mean_tpr-std_tpr,0), np.minimum(mean_tpr+std_tpr,1),
                     color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity'); plt.ylabel('Sensitivity')
    plt.title(title); plt.legend(loc='lower right')
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")


def plot_confusion(ground_truth_list, predictions_list, filename=None):
    cm = np.zeros((2,2), dtype=int)
    for g, p in zip(ground_truth_list, predictions_list):
        cm += confusion_matrix(g, p.round())
    plt.figure(figsize=(6,5)); plt.gca().grid(False)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    tick_marks = [0,1]
    plt.xticks(tick_marks, ['survive','die']); plt.yticks(tick_marks, ['survive','die'])
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(2),range(2)):
        plt.text(j, i, str(cm[i,j]), ha='center',
                 color='white' if cm[i,j] > thresh else 'black')
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.close()


# ── training ──────────────────────────────────────────────────────────────────

def train():
    # ── load data ────────────────────────────────────────────────────────────
    input_df, labels_df = build_dataset()

    n_patients    = input_df.index.get_level_values('subject_id').nunique()
    n_predictors  = input_df.index.get_level_values('measurement_name').nunique()
    print(f"Patients: {n_patients}, Predictors: {n_predictors}, Hours: {input_df.shape[1]}")

    X = input_df.values.reshape((n_patients, n_predictors, 48))
    X = X.reshape((-1, 1, n_predictors, 48)).astype(np.float32)
    y = labels_df['mort_icu'].values[:, None].astype(np.float32)
    print(f"X shape: {X.shape}, y shape: {y.shape}, mortality rate: {y.mean():.3f}")

    # ── cross-validation ─────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_idx = list(cv.split(X=np.zeros(n_patients), y=y.ravel()))

    subject_ids = input_df.index.get_level_values('subject_id').unique()
    fold_mean, fold_std = [], []
    for train_idx, _ in cv_idx:
        train_subjects = subject_ids[train_idx]
        fold_data = input_df.loc[train_subjects]
        m = fold_data.groupby(level='measurement_name').mean().mean(axis=1).values[:, None]
        s = fold_data.groupby(level='measurement_name').std().mean(axis=1).values[:, None]
        fold_mean.append(m)
        fold_std.append(s)

    os.makedirs('plots', exist_ok=True)

    predictions, ground_truth, fold_history = [], [], []

    batch_size   = 32
    epochs       = 50
    class_weight = {0: 1, 1: 10}

    for i, (train_idx, val_idx) in enumerate(cv_idx):
        print(f"\nFold {i}  |  train={len(train_idx)}  val={len(val_idx)}  "
              f"pos_val={int(y[val_idx].sum())}")

        ground_truth.append(y[val_idx])

        model = create_model((1, n_predictors, 48))
        sgd = optimizers.SGD(learning_rate=0.01, decay=1e-7, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        Xtr = np.nan_to_num((X[train_idx] - fold_mean[i]) / (fold_std[i] + 1e-8))
        Xvl = np.nan_to_num((X[val_idx]   - fold_mean[i]) / (fold_std[i] + 1e-8))

        ckpt = ModelCheckpoint(f'plots/kfold{i}_best.keras', monitor='val_loss', save_best_only=True)
        hist = model.fit(Xtr, y[train_idx], epochs=epochs, batch_size=batch_size,
                         class_weight=class_weight, callbacks=[ckpt],
                         validation_data=(Xvl, y[val_idx]), verbose=1)
        fold_history.append(hist)

        from tensorflow.keras.models import load_model
        model = load_model(f'plots/kfold{i}_best.keras')
        predictions.append(model.predict(Xvl, verbose=1))

    # ── plots ─────────────────────────────────────────────────────────────────
    # Loss curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for k, h in enumerate(fold_history):
        j = np.argmin(h.history['val_loss'])
        ax = axes[k//3, k%3]
        ax.plot(h.history['loss'], label='train')
        ax.plot(h.history['val_loss'], label='validation')
        ax.set_title(f'Fold {k} (best={h.history["val_loss"][j]:.3f}, epoch {j})')
        ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend()
    plt.tight_layout()
    plt.savefig('plots/loss_history.png', dpi=150)
    plt.close()

    cross_validation_plot_roc(ground_truth, predictions,
                              'Receiver operating characteristic curve',
                              filename='plots/roc_curve_validation.png')
    plot_confusion(ground_truth, predictions, filename='plots/confusion_matrix.png')

    print("\nSaved plots to plots/")


if __name__ == '__main__':
    train()
