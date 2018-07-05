from __future__ import print_function
import time

import tensorflow as tf

from six.moves import xrange


def model(x, nlogits, train=False):
  b = tf.shape(x)[0]

  if args.infer_legacy:
    window = tf.contrib.signal.hann_window(1024)
    windows = []
    for i in xrange(0, 16384, 128):
      x_window = x[:, i:i+1024]
      x_padded = tf.pad(x_window, [[0, 0], [0, max(0, i + 1024 - 16384)]])
      x_windowed = x_padded * window
      windows.append(x_windowed)
    windows = tf.stack(windows, axis=1)
    X = tf.spectral.rfft(windows)
  else:
    X = tf.contrib.signal.stft(x, 1024, 128, pad_end=True)

  X_mag = tf.abs(X)

  W = tf.contrib.signal.linear_to_mel_weight_matrix(
      num_mel_bins=128,
      num_spectrogram_bins=513,
      sample_rate=16000,
      lower_edge_hertz=40.,
      upper_edge_hertz=7800.,
      dtype=tf.float32)

  X_mag = tf.reshape(X_mag, [-1, 513])
  X_mel = tf.matmul(X_mag, W)
  X_mel = tf.reshape(X_mel, [b, 128, 128])
  X_lmel = tf.log(X_mel + 1e-6)

  x = tf.stop_gradient(X_lmel)

  dropout = 0.5 if train else 0.

  x = tf.layers.batch_normalization(x, training=train)
  x = tf.expand_dims(x, axis=3)
  x = tf.layers.conv2d(x, 128, (5, 5), padding='same', activation=tf.nn.relu)
  x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

  x = tf.layers.batch_normalization(x, training=train)
  x = tf.layers.conv2d(x, 128, (5, 5), padding='same', activation=tf.nn.relu)
  x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

  x = tf.layers.batch_normalization(x, training=train)
  x = tf.layers.conv2d(x, 128, (5, 5), padding='same', activation=tf.nn.relu)
  x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

  x = tf.layers.batch_normalization(x, training=train)
  x = tf.layers.conv2d(x, 128, (5, 5), padding='same', activation=tf.nn.relu)
  x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

  x = tf.reshape(x, [b, 8 * 8 * 128])
  x = tf.layers.batch_normalization(x, training=train)

  x = tf.layers.dense(x, nlogits)

  if train:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    assert len(update_ops) == 10
    with tf.control_dependencies(update_ops):
      x = tf.identity(x)

  return x


def record_to_xy(example_proto, labels):
  features = {
      'samples': tf.FixedLenSequenceFeature([1], tf.float32, allow_missing=True),
      'label': tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)
  }
  example = tf.parse_single_example(example_proto, features)

  wav = example['samples'][:, 0]
  wav = wav[:16384]
  wav = tf.pad(wav, [[0, 16384 - tf.shape(wav)[0]]])
  wav.set_shape([16384])

  label_chars = example['label']
  # Truncate labels for TIMIT
  label_lens = [len(l) for l in labels]
  if len(set(label_lens)) == 1:
    label_chars = label_chars[:label_lens[0]]

  label = tf.reduce_join(label_chars, 0)
  label_id = tf.constant(0, dtype=tf.int32)
  nmatches = tf.constant(0)

  for i, label_candidate in enumerate(labels):
    match = tf.cast(tf.equal(label, label_candidate), tf.int32)
    label_id += i * match
    nmatches += match

  with tf.control_dependencies([tf.assert_equal(nmatches, 1)]):
    return wav, label_id


def eval(fps, args):
  import numpy as np

  eval_dir = os.path.join(args.train_dir, 'eval_{}'.format(args.eval_split))
  if not os.path.isdir(eval_dir):
    os.makedirs(eval_dir)

  with tf.name_scope('eval_loader'):
    dataset = tf.data.TFRecordDataset(fps)
    dataset = dataset.map(lambda x: record_to_xy(x, args.data_labels))
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(args.eval_batch_size))
    iterator = dataset.make_one_shot_iterator()

    x, y = iterator.get_next()

  with tf.variable_scope('classifier'):
    logits = model(x, len(args.data_labels))

  xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
  xent_mean = tf.reduce_mean(xent)

  preds = tf.argmax(logits, axis=1, output_type=tf.int32)
  acc = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))
  step = tf.train.get_or_create_global_step()

  summaries = [
      tf.summary.scalar('acc', acc),
      tf.summary.scalar('xent', xent_mean)
  ]
  summaries = tf.summary.merge(summaries)
  summary_writer = tf.summary.FileWriter(eval_dir)

  saver = tf.train.Saver(max_to_keep=1)

  def eval_ckpt_fp(sess, ckpt_fp):
    saver.restore(sess, ckpt_fp)

    _xents = []
    _accs = []
    while True:
      try:
        _xent, _acc = sess.run([xent_mean, acc])
      except:
        break
      _xents.append(_xent)
      _accs.append(_acc)

    _step = sess.run(step)

    _summaries = sess.run(summaries, {acc: np.mean(_accs), xent_mean: np.mean(_xents)})
    summary_writer.add_summary(_summaries, _step)

    return _step, np.mean(_accs)

  if args.eval_ckpt_fp is not None:
    # Eval one
    with tf.Session() as sess:
      eval_ckpt_fp(sess, args.eval_ckpt_fp)
  else:
    # Loop, waiting for checkpoints
    ckpt_fp = None
    _best_acc = 0.
    while True:
      latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
      if latest_ckpt_fp != ckpt_fp:
        print('Preview: {}'.format(latest_ckpt_fp))

        with tf.Session() as sess:
          _step, _acc = eval_ckpt_fp(sess, latest_ckpt_fp)
          if _acc > _best_acc:
            saver.save(sess, os.path.join(eval_dir, 'best_acc'), _step)
            _best_acc = _acc

        print('Done')

        ckpt_fp = latest_ckpt_fp

      time.sleep(1)


def infer(args):
  import cPickle as pickle

  infer_dir = os.path.join(args.train_dir, 'infer')
  if not os.path.isdir(infer_dir):
    os.makedirs(infer_dir)

  # Placeholders for sampling stage
  x = tf.placeholder(tf.float32, [None, 16384], name='x')
  labels = tf.constant(args.data_labels, name='labels')

  with tf.variable_scope('classifier'):
    logits = model(x, len(args.data_labels))

  scores = tf.nn.softmax(logits, name='scores')
  pred = tf.argmax(logits, axis=1, output_type=tf.int32, name='pred')
  pred_label = tf.gather(labels, pred, name='pred_label')

  # Create saver
  all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  global_step = tf.train.get_or_create_global_step()
  saver = tf.train.Saver(all_vars + [global_step])

  # Export graph
  tf.train.write_graph(tf.get_default_graph(), infer_dir, 'infer.pbtxt')

  # Export MetaGraph
  infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
  tf.train.export_meta_graph(
      filename=infer_metagraph_fp,
      clear_devices=True,
      saver_def=saver.as_saver_def())

  # Reset graph (in case training afterwards)
  tf.reset_default_graph()


def train(fps, args):
  with tf.name_scope('loader'):
    dataset = tf.data.TFRecordDataset(fps)
    dataset = dataset.map(lambda x: record_to_xy(x, args.data_labels))
    dataset = dataset.shuffle(buffer_size=8192)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(args.train_batch_size))
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()

    x, y = iterator.get_next()

  with tf.variable_scope('classifier'):
    logits = model(x, len(args.data_labels), train=True)
  for v in tf.global_variables():
    print(v.get_shape(), v.name)

  xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
  xent_mean = tf.reduce_mean(xent)

  opt = tf.train.GradientDescentOptimizer(1e-4)
  train_op = opt.minimize(xent_mean, global_step=tf.train.get_or_create_global_step())

  preds = tf.argmax(logits, axis=1, output_type=tf.int32)
  acc = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))

  tf.summary.audio('x', tf.expand_dims(x, axis=2), 16000)
  tf.summary.scalar('xent', xent_mean)
  tf.summary.scalar('acc', acc)
  tf.summary.histogram('xent', xent_mean)

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_save_secs,
      save_summaries_secs=args.train_summary_secs) as sess:
    while True:
      _, _acc = sess.run([train_op, acc])


if __name__ == '__main__':
  import argparse
  import glob
  import os
  import sys

  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'eval'])
  parser.add_argument('train_dir', type=str,
      help='Training directory')

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_dir', type=str,
      help='Data directory')
  data_args.add_argument('--data_labels', type=str,
      help='Comma-separated list of labels')

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_batch_size', type=int,
      help='Batch size')
  train_args.add_argument('--train_save_secs', type=int,
      help='How often to save model')
  train_args.add_argument('--train_summary_secs', type=int,
      help='How often to report summaries')

  eval_args = parser.add_argument_group('Eval')
  eval_args.add_argument('--eval_batch_size', type=int,
      help='Batch size')
  eval_args.add_argument('--eval_split', type=str,
      help='Eval split')
  eval_args.add_argument('--eval_ckpt_fp', type=str,
      help='If set, evaluate this checkpoint once')

  infer_args = parser.add_argument_group('Infer')
  infer_args.add_argument('--infer_legacy', action='store_true', dest='infer_legacy',
      help='If set, create graph compatible with tf1.1')

  parser.set_defaults(
    data_dir=None,
    data_labels=None,
    train_batch_size=64,
    train_save_secs=300,
    train_summary_secs=120,
    eval_batch_size=64,
    eval_split='valid',
    eval_ckpt_fp=None,
    infer_legacy=False)

  args = parser.parse_args()

  labels = [l.strip() for l in args.data_labels.split(',')]
  setattr(args, 'data_labels', labels)

  # Make train dir
  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  # Save args
  with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

  # Assign appropriate split for mode
  if args.mode == 'train':
    split = 'train'
  elif args.mode == 'eval':
    split = args.eval_split
  else:
    raise NotImplementedError()

  # Find group fps and make splits
  fps = glob.glob(os.path.join(args.data_dir, split) + '*.tfrecord')

  if args.mode == 'train':
    infer(args)
    train(fps, args)
  if args.mode == 'eval':
    eval(fps, args)
  else:
    raise NotImplementedError()
