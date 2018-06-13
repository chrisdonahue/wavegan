from __future__ import print_function
import numpy as np
import tensorflow as tf

from six.moves import xrange


def inception_score(
    audio_fps,
    k,
    metagraph_fp,
    ckpt_fp,
    batch_size=100,
    tf_ffmpeg_ext=None,
    fix_length=False):
  use_tf_ffmpeg = tf_ffmpeg_ext is not None
  if not use_tf_ffmpeg:
    from scipy.io.wavfile import read as wavread

  if len(audio_fps) % k != 0:
    raise Exception('Number of audio files ({}) is not divisible by k ({})'.format(len(audio_fps), k))
  group_size = len(audio_fps) // k

  # Restore graph
  graph = tf.Graph()
  with graph.as_default():
    saver = tf.train.import_meta_graph(metagraph_fp)
    
    if use_tf_ffmpeg:
      x_fp = tf.placeholder(tf.string, [])
      x_bin = tf.read_file(x_fp)
      x_samps = tf.contrib.ffmpeg.decode_audio(x_bin, tf_ffmpeg_ext, 16000, 1)[:, 0]
  x = graph.get_tensor_by_name('x:0')
  scores = graph.get_tensor_by_name('scores:0')

  # Restore weights
  sess = tf.Session(graph=graph)
  saver.restore(sess, ckpt_fp)

  # Evaluate audio
  _all_scores = []
  for i in xrange(0, len(audio_fps), batch_size):
    batch = audio_fps[i:i+batch_size]

    # Load audio files
    _xs = []
    for audio_fp in batch:
      if use_tf_ffmpeg:
        _x = sess.run(x_samps, {x_fp: audio_fp})
      else:
        fs, _x = wavread(audio_fp)
        if fs != 16000:
          raise Exception('Invalid sample rate ({})'.format(fs))
        _x = _x.astype(np.float32)
        _x /= 32767.

      if _x.ndim != 1:
        raise Exception('Invalid shape ({})'.format(_x.shape))

      if fix_length:
        _x = _x[:16384]
        #_x = _x[-16384:]
        _x = np.pad(_x, (0, 16384 - _x.shape[0]), 'constant')

      if _x.shape[0] != 16384:
        raise Exception('Invalid number of samples ({})'.format(_x.shape[0]))

      _xs.append(_x)

    # Compute model scores
    _all_scores.append(sess.run(scores, {x: _xs}))

  sess.close()

  # Find labels
  _all_scores = np.concatenate(_all_scores, axis=0)
  _all_labels = np.argmax(_all_scores, axis=1)

  # Compute inception scores
  _inception_scores = []
  for i in xrange(k):
    _group = _all_scores[i * group_size:(i + 1) * group_size]
    _kl = _group * (np.log(_group) - np.log(np.expand_dims(np.mean(_group, 0), 0)))
    _kl = np.mean(np.sum(_kl, 1))
    _inception_scores.append(np.exp(_kl))

  return np.mean(_inception_scores), np.std(_inception_scores), _all_labels


if __name__ == '__main__':
  import argparse
  import glob
  import os
  import random
  import sys

  parser = argparse.ArgumentParser()
  parser.add_argument('--audio_dir', type=str,
      help='Directory with 16-bit signed integer PCM WAV files at 16kHz')
  parser.add_argument('--fix_length', action='store_true', dest='fix_length',
      help='If set, pad or trim audio files to length 16384')
  parser.add_argument('--labels_fp', type=str,
      help='If set, write model predictions to this file')
  parser.add_argument('--metagraph_fp', type=str,
      help='MetaGraph for classifier; must have tensors x:0 [None, 16384] and scores:0 [None, 10]')
  parser.add_argument('--ckpt_fp', type=str,
      help='Checkpoint for metagraph')
  parser.add_argument('--n', type=int,
      help='Number of samples to test')
  parser.add_argument('--k', type=int,
      help='Number of subsets to score')
  parser.add_argument('--batch_size', type=int,
      help='Evaluate audio in batches of this size')
  parser.add_argument('--tf_ffmpeg_ext', type=str,
      help='If set, uses ffmpeg to decode audio files with specified extension through tensorflow')

  parser.set_defaults(
    audio_dir=None,
    fix_length=False,
    labels_fp=None,
    metagraph_fp='infer.meta',
    ckpt_fp='best_acc-103005',
    n=50000,
    k=10,
    batch_size=100,
    tf_ffmpeg_ext=None)

  args = parser.parse_args()

  # Find audio files
  if args.audio_dir is None:
    raise Exception('No audio directory specified')
  ext = 'wav' if args.tf_ffmpeg_ext is None else args.tf_ffmpeg_ext
  audio_fps = sorted(glob.glob(os.path.join(args.audio_dir, '*.{}'.format(ext))))
  random.seed(0)
  random.shuffle(audio_fps)
  if len(audio_fps) < args.n:
    raise Exception('Found fewer ({}) than specified ({}) audio files'.format(len(audio_fps), args.n))
  audio_fps = audio_fps[:args.n]

  # Compute scores
  mean, std, labels = inception_score(
      audio_fps,
      args.k,
      args.metagraph_fp,
      args.ckpt_fp,
      batch_size=args.batch_size,
      tf_ffmpeg_ext=args.tf_ffmpeg_ext,
      fix_length=args.fix_length)
  print('Inception score: {} +- {}'.format(mean, std))

  print('p(y)')
  for i in xrange(10):
    n = len(filter(lambda x: x == i, labels))
    print('{}: {}'.format(i, n / float(args.n)))

  # Save labels
  if args.labels_fp is not None:
    labels_txt = []
    for audio_fp, label in zip(audio_fps, labels):
      labels_txt.append(','.join([audio_fp, str(label)]))
    with open(args.labels_fp, 'w') as f:
      f.write('\n'.join(labels_txt))
