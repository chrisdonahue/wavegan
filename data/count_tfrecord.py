from __future__ import print_function
from collections import Counter
import os
import sys

import tensorflow as tf

tfrecord_fps = sys.argv[1:]

def _mapper(example_proto):
  features = {
      'samples': tf.FixedLenSequenceFeature([1], tf.float32, allow_missing=True),
      'label': tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)
  }
  example = tf.parse_single_example(example_proto, features)

  wav = example['samples']
  wav_len = tf.shape(wav)[0]

  label = tf.reduce_join(example['label'], 0)

  return wav_len, label

dataset = tf.data.TFRecordDataset(tfrecord_fps)
dataset = dataset.map(_mapper)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(1))
xs, ys = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
  n = 0
  nsamps = 0
  label_counts = Counter()
  while True:
    try:
      _xs, _ys = sess.run([xs, ys])
      label_counts[_ys[0]] += 1
      n += 1
      nsamps += _xs[0]
    except:
      break

print('-' * 80)
print('Number of examples: {}'.format(n))
print('Number of audio samples: {}'.format(nsamps))
print('Number of seconds: {}'.format(nsamps / 16000.))
print('Number of hours: {}'.format(nsamps / 16000. / 3600.))

if len(label_counts) > 1:
  print('-' * 80)
  print('Labels')
  for label, count in label_counts.items():
    print('{}, count {}, proportion {}'.format(label, count, count / float(n)))
