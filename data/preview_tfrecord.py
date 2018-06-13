from __future__ import print_function
batch_size = 8
_WINDOW_LEN = 16384

import os
import sys

from scipy.io.wavfile import write as wavwrite
import tensorflow as tf

tfrecord_fp, out_dir = sys.argv[1:]

if not os.is_dir(out_dir):
  os.makedirs(out_dir)

def _mapper(example_proto):
  features = {
      'id': tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
      'label': tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
      'slice': tf.FixedLenFeature([], tf.int64),
      'samples': tf.FixedLenSequenceFeature([1], tf.float32, allow_missing=True)
  }
  example = tf.parse_single_example(example_proto, features)

  wav_id = tf.reduce_join(example['id'], 0)
  wav_label = tf.reduce_join(example['label'], 0)
  wav_slice = example['slice']
  wav = example['samples']
  wav_len = tf.shape(wav)[0]

  start_max = wav_len - _WINDOW_LEN
  start_max = tf.maximum(start_max, 0)

  start = tf.random_uniform([], maxval=start_max + 1, dtype=tf.int32)

  wav = wav[start:start+_WINDOW_LEN]

  wav = tf.pad(wav, [[0, _WINDOW_LEN - tf.shape(wav)[0]], [0, 0]])

  wav.set_shape([_WINDOW_LEN, 1])

  return wav_id, wav_label, wav_slice, wav, wav_len

dataset = tf.data.TFRecordDataset([tfrecord_fp])
dataset = dataset.map(_mapper)
dataset = dataset.shuffle(buffer_size=batch_size)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
xs = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
  _xs = sess.run(xs)

  for i, (_wav_id, _wav_label, _wav_slice, _wav, _wav_len) in enumerate(zip(*_xs)):
    print('-' * 80)
    print(i)
    print('ID: {}'.format(_wav_id))
    print('Label: {}'.format(_wav_label))
    print('Slice #: {}'.format(_wav_slice))
    print('Len: {}'.format(_wav_len))

    out_fp = os.path.join(out_dir, '{}.wav'.format(str(i).zfill(2)))
    wavwrite(out_fp, 16000, _wav)
