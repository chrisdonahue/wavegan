from __future__ import print_function
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import time


def nn_dist(train_set, query_set, exclude_self):
  # Flatten
  train_set = np.reshape(train_set, [train_set.shape[0], -1])
  query_set = np.reshape(query_set, [query_set.shape[0], -1])

  # Create and query model
  print('Creating model')
  start = time.time()
  model = NearestNeighbors(n_neighbors=2 if exclude_self else 1, algorithm='ball_tree').fit(train_set)
  print('Took {} seconds'.format(time.time() - start))

  print('Querying model')
  start = time.time()
  dists, _ = model.kneighbors(query_set)
  print('Took {} seconds'.format(time.time() - start))

  # If specified, exclude first nearest neighbor (duplicate) if it is nonzero
  if exclude_self:
    dists_excluded = []
    for dist0, dist1 in dists:
      if dist0 == 0:
        dists_excluded.append(dist1)
      else:
        dists_excluded.append(dist0)
    dists = dists_excluded
  else:
    dists = dists[:, 0]

  return np.mean(dists), np.std(dists)


if __name__ == '__main__':
  import argparse
  import cPickle as pickle
  import os
  import random
  import sys

  parser = argparse.ArgumentParser()
  parser.add_argument('--train_set', type=str,
      help='Set to train KNN model')
  parser.add_argument('--query_set', type=str,
      help='Query set for KNN model')

  args = parser.parse_args()

  with open(args.train_set, 'rb') as f:
    train_set = pickle.load(f)
  with open(args.query_set, 'rb') as f:
    query_set = pickle.load(f)

  mean, std = nn_dist(train_set, query_set, args.train_set == args.query_set)
  print('Similarity: {} +- {}'.format(mean, std))
