#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import math

EMBEDDING_SIZE = 128  # Dimension of the embedding vector.
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
NUM_SAMPLED = 64    # Number of negative examples to sample.

# valid_examples = np.array(random.sample(np.arange(VALID_WINDOW), VALID_SIZE))
# print("valid_examples", valid_examples)

def make_graph(vocabulary_size, batch_size, valid_examples):
    graph = tf.Graph()
    with graph.as_default():
        # CPUでないと動かない。
        with graph.device("/cpu:0"):
            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            
            # Construct the variables.
            
            # input embeddings: W_I
            embeddings = tf.Variable(
                tf.random_uniform(
                    [vocabulary_size, EMBEDDING_SIZE], 
                    -1.0, 
                    1.0
                )
            )
            
            # output weights: W_O
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [vocabulary_size, EMBEDDING_SIZE],
                    stddev=1.0 / math.sqrt(EMBEDDING_SIZE)
                )
            )
            
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
                
            # Look up embeddings for inputs. v_t = W_I x_t
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    nce_weights, # W_O
                    nce_biases,  # b_O
                    embed,  # v_t
                    train_labels,
                    NUM_SAMPLED, # the number of classes to randomly sample per batch: Negative sampling 
                    vocabulary_size, # the number of possible classes.
                    num_true=1 # the number of target classes per training example
                )
            )
            
            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
            
            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm # make v_t unit vector
            
            # calculate unit vector v_t = W_I x_t
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, 
                valid_dataset
            )
            
            # これは何をしているのか？
            # |v_t|^2
            similarity = tf.matmul(
                valid_embeddings, 
                normalized_embeddings, 
                transpose_b=True
            )
            
            print("valid_embeddings._shape: ", valid_embeddings._shape)
            print("normalized_embeddings._shape: ", normalized_embeddings._shape)
            print("similarity._shape: ", similarity._shape)
    return graph, train_inputs, train_labels, valid_dataset, optimizer, loss, similarity, normalized_embeddings


if __name__ == "__main__":
    pass


