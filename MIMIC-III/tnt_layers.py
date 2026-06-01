import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class TreeEmbeddingLayer(layers.Layer):
    """
    Keras implementation of the TNT (Tree-to-Neural-Network) embedding logic.
    Transforms leaf indices from an ensemble into learned feature embeddings.
    """
    def __init__(self, num_trees, max_leaves_per_tree, embedding_dim, seed=42, **kwargs):
        super(TreeEmbeddingLayer, self).__init__(**kwargs)
        self.num_trees = num_trees
        self.max_leaves_per_tree = max_leaves_per_tree
        self.embedding_dim = embedding_dim
        # Vocabulary covers all possible leaf IDs across all trees using offsets
        self.total_vocabulary = num_trees * max_leaves_per_tree

        # Tree permutation (fixed)
        rng = np.random.RandomState(seed)
        self.tree_perm = rng.permutation(num_trees)
        
        # Offset to ensure disjoint embedding ranges for every tree
        self.tree_offsets = np.arange(self.num_trees) * self.max_leaves_per_tree
        self.tree_offsets = tf.constant(self.tree_offsets, dtype=tf.int32)

    def build(self, input_shape):
        # Single embedding table for all trees
        self.embedding = layers.Embedding(
            self.total_vocabulary, 
            self.embedding_dim, 
            name="tree_leaf_embeddings"
        )
        super(TreeEmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, 48, num_trees)
        # Permute trees for stochasticity (consistent with TNT paper)
        x = tf.gather(inputs, self.tree_perm, axis=-1)
        
        # Apply offsets: (batch, 48, num_trees)
        x = tf.cast(x, tf.int32) + self.tree_offsets
        
        # Embed: (batch, 48, num_trees, embedding_dim)
        embeddings = self.embedding(x)
        
        # Concatenate embeddings for each hour: (batch, 48, num_trees * embedding_dim)
        return tf.reshape(embeddings, (-1, 48, self.num_trees * self.embedding_dim))