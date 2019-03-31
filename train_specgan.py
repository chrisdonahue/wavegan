from __future__ import print_function

try:
  import cPickle as pickle
except:
  import pickle
from functools import reduce
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import xrange

import loader
from specgan import SpecGANGenerator, SpecGANDiscriminator


"""
  Constants
"""
# TODO: Support different generation (slice) lengths in SpecGAN.
_SLICE_LEN = 16384
_CLIP_NSTD = 3.
_LOG_EPS = 1e-6


"""
  Convert raw audio to spectrogram
"""
def t_to_f(x, X_mean, X_std):
  x = x[:, :, 0]
  X = tf.contrib.signal.stft(x, 256, 128, pad_end=True)
  X = X[:, :, :-1]

  X_mag = tf.abs(X)
  X_lmag = tf.log(X_mag + _LOG_EPS)
  X_norm = (X_lmag - X_mean[:-1]) / X_std[:-1]
  X_norm /= _CLIP_NSTD
  X_norm = tf.clip_by_value(X_norm, -1., 1.)
  X_norm = tf.expand_dims(X_norm, axis=3)

  X_norm = tf.stop_gradient(X_norm)

  return X_norm


"""
  Griffin-Lim phase estimation from magnitude spectrum
"""
def invert_spectra_griffin_lim(X_mag, nfft, nhop, ngl):
    X = tf.complex(X_mag, tf.zeros_like(X_mag))

    def b(i, X_best):
        x = tf.contrib.signal.inverse_stft(X_best, nfft, nhop)
        X_est = tf.contrib.signal.stft(x, nfft, nhop)
        phase = X_est / tf.cast(tf.maximum(1e-8, tf.abs(X_est)), tf.complex64)
        X_best = X * phase
        return i + 1, X_best

    i = tf.constant(0)
    c = lambda i, _: tf.less(i, ngl)
    _, X = tf.while_loop(c, b, [i, X], back_prop=False)

    x = tf.contrib.signal.inverse_stft(X, nfft, nhop)
    x = x[:, :_SLICE_LEN]

    return x


"""
  Estimate raw audio for spectrogram
"""
def f_to_t(X_norm, X_mean, X_std, ngl=16):
  X_norm = X_norm[:, :, :, 0]
  X_norm = tf.pad(X_norm, [[0,0], [0,0], [0,1]])
  X_norm *= _CLIP_NSTD
  X_lmag = (X_norm * X_std) + X_mean
  X_mag = tf.exp(X_lmag)

  x = invert_spectra_griffin_lim(X_mag, 256, 128, ngl)
  x = tf.reshape(x, [-1, _SLICE_LEN, 1])

  return x


"""
  Render normalized spectrogram as uint8 image
"""
def f_to_img(X_norm):
  X_uint8 = X_norm + 1.
  X_uint8 *= 128.
  X_uint8 = tf.clip_by_value(X_uint8, 0., 255.)
  X_uint8 = tf.cast(X_uint8, tf.uint8)

  X_uint8 = tf.map_fn(lambda x: tf.image.rot90(x, 1), X_uint8)

  return X_uint8


"""
  Trains a SpecGAN
"""
def train(fps, args):
  with tf.name_scope('loader'):
    x_wav = loader.decode_extract_and_batch(
        fps,
        batch_size=args.train_batch_size,
        slice_len=_SLICE_LEN,
        decode_fs=args.data_sample_rate,
        decode_num_channels=1,
        decode_fast_wav=args.data_fast_wav,
        decode_parallel_calls=4,
        slice_randomize_offset=False if args.data_first_slice else True,
        slice_first_only=args.data_first_slice,
        slice_overlap_ratio=0. if args.data_first_slice else args.data_overlap_ratio,
        slice_pad_end=True if args.data_first_slice else args.data_pad_end,
        repeat=True,
        shuffle=True,
        shuffle_buffer_size=4096,
        prefetch_size=args.train_batch_size * 4,
        prefetch_gpu_num=args.data_prefetch_gpu_num)[:, :, 0]

    x = t_to_f(x_wav, args.data_moments_mean, args.data_moments_std)

  # Make z vector
  z = tf.random_uniform([args.train_batch_size, args.specgan_latent_dim], -1., 1., dtype=tf.float32)

  # Make generator
  with tf.variable_scope('G'):
    G_z = SpecGANGenerator(z, train=True, **args.specgan_g_kwargs)
  G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

  # Print G summary
  print('-' * 80)
  print('Generator vars')
  nparams = 0
  for v in G_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

  # Summarize
  x_gl = f_to_t(x, args.data_moments_mean, args.data_moments_std, args.specgan_ngl)
  G_z_gl = f_to_t(G_z, args.data_moments_mean, args.data_moments_std, args.specgan_ngl)
  tf.summary.audio('x_wav', x_wav, args.data_sample_rate)
  tf.summary.audio('x', x_gl, args.data_sample_rate)
  tf.summary.audio('G_z', G_z_gl, args.data_sample_rate)
  G_z_rms = tf.sqrt(tf.reduce_mean(tf.square(G_z_gl[:, :, 0]), axis=1))
  x_rms = tf.sqrt(tf.reduce_mean(tf.square(x_gl[:, :, 0]), axis=1))
  tf.summary.histogram('x_rms_batch', x_rms)
  tf.summary.histogram('G_z_rms_batch', G_z_rms)
  tf.summary.scalar('x_rms', tf.reduce_mean(x_rms))
  tf.summary.scalar('G_z_rms', tf.reduce_mean(G_z_rms))
  tf.summary.image('x', f_to_img(x))
  tf.summary.image('G_z', f_to_img(G_z))

  # Make real discriminator
  with tf.name_scope('D_x'), tf.variable_scope('D'):
    D_x = SpecGANDiscriminator(x, **args.specgan_d_kwargs)
  D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

  # Print D summary
  print('-' * 80)
  print('Discriminator vars')
  nparams = 0
  for v in D_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
  print('-' * 80)

  # Make fake discriminator
  with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
    D_G_z = SpecGANDiscriminator(G_z, **args.specgan_d_kwargs)

  # Create loss
  D_clip_weights = None
  if args.specgan_loss == 'dcgan':
    fake = tf.zeros([args.train_batch_size], dtype=tf.float32)
    real = tf.ones([args.train_batch_size], dtype=tf.float32)

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_G_z,
      labels=real
    ))

    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_G_z,
      labels=fake
    ))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_x,
      labels=real
    ))

    D_loss /= 2.
  elif args.specgan_loss == 'lsgan':
    G_loss = tf.reduce_mean((D_G_z - 1.) ** 2)
    D_loss = tf.reduce_mean((D_x - 1.) ** 2)
    D_loss += tf.reduce_mean(D_G_z ** 2)
    D_loss /= 2.
  elif args.specgan_loss == 'wgan':
    G_loss = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

    with tf.name_scope('D_clip_weights'):
      clip_ops = []
      for var in D_vars:
        clip_bounds = [-.01, .01]
        clip_ops.append(
          tf.assign(
            var,
            tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
          )
        )
      D_clip_weights = tf.group(*clip_ops)
  elif args.specgan_loss == 'wgan-gp':
    G_loss = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

    alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1, 1], minval=0., maxval=1.)
    differences = G_z - x
    interpolates = x + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
      D_interp = SpecGANDiscriminator(interpolates, **args.specgan_d_kwargs)

    LAMBDA = 10
    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    D_loss += LAMBDA * gradient_penalty
  else:
    raise NotImplementedError()

  tf.summary.scalar('G_loss', G_loss)
  tf.summary.scalar('D_loss', D_loss)

  # Create (recommended) optimizer
  if args.specgan_loss == 'dcgan':
    G_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5)
    D_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5)
  elif args.specgan_loss == 'lsgan':
    G_opt = tf.train.RMSPropOptimizer(
        learning_rate=1e-4)
    D_opt = tf.train.RMSPropOptimizer(
        learning_rate=1e-4)
  elif args.specgan_loss == 'wgan':
    G_opt = tf.train.RMSPropOptimizer(
        learning_rate=5e-5)
    D_opt = tf.train.RMSPropOptimizer(
        learning_rate=5e-5)
  elif args.specgan_loss == 'wgan-gp':
    G_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9)
    D_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9)
  else:
    raise NotImplementedError()

  # Create training ops
  G_train_op = G_opt.minimize(G_loss, var_list=G_vars,
      global_step=tf.train.get_or_create_global_step())
  D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

  # Run training
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_save_secs,
      save_summaries_secs=args.train_summary_secs) as sess:
    print('-' * 80)
    print('Training has started. Please use \'tensorboard --logdir={}\' to monitor.'.format(args.train_dir))
    while True:
      # Train discriminator
      for i in xrange(args.specgan_disc_nupdates):
        sess.run(D_train_op)

        # Enforce Lipschitz constraint for WGAN
        if D_clip_weights is not None:
          sess.run(D_clip_weights)

      # Train generator
      sess.run(G_train_op)


"""
  Creates and saves a MetaGraphDef for simple inference
  Tensors:
    'samp_z_n' int32 []: Sample this many latent vectors
    'samp_z' float32 [samp_z_n, 100]: Resultant latent vectors
    'z:0' float32 [None, 100]: Input latent vectors
    'ngl:0' int32 []: Number of Griffin-Lim iterations for resynthesis
    'flat_pad:0' int32 []: Number of padding samples to use when flattening batch to a single audio file
    'G_z_norm:0' float32 [None, 128, 128, 1]: Generated outputs (frequency domain)
    'G_z:0' float32 [None, 16384, 1]: Generated outputs (Griffin-Lim'd to time domain)
    'G_z_norm_uint8:0' uint8 [None, 128, 128, 1]: Preview spectrogram image
    'G_z_int16:0' int16 [None, 16384, 1]: Same as above but quantizied to 16-bit PCM samples
    'G_z_flat:0' float32 [None, 1]: Outputs flattened into single audio file
    'G_z_flat_int16:0' int16 [None, 1]: Same as above but quantized to 16-bit PCM samples
  Example usage:
    import tensorflow as tf
    tf.reset_default_graph()

    saver = tf.train.import_meta_graph('infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'model.ckpt-10000')

    z_n = graph.get_tensor_by_name('samp_z_n:0')
    _z = sess.run(graph.get_tensor_by_name('samp_z:0'), {z_n: 10})

    z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z})
"""
def infer(args):
  infer_dir = os.path.join(args.train_dir, 'infer')
  if not os.path.isdir(infer_dir):
    os.makedirs(infer_dir)

  # Subgraph that generates latent vectors
  samp_z_n = tf.placeholder(tf.int32, [], name='samp_z_n')
  samp_z = tf.random_uniform([samp_z_n, args.specgan_latent_dim], -1.0, 1.0, dtype=tf.float32, name='samp_z')

  # Input zo
  z = tf.placeholder(tf.float32, [None, args.specgan_latent_dim], name='z')
  ngl = tf.placeholder(tf.int32, [], name='ngl')
  flat_pad = tf.placeholder(tf.int32, [], name='flat_pad')

  # Execute generator
  with tf.variable_scope('G'):
    G_z_norm = SpecGANGenerator(z, train=False, **args.specgan_g_kwargs)
  G_z_norm = tf.identity(G_z_norm, name='G_z_norm')
  G_z = f_to_t(G_z_norm, args.data_moments_mean, args.data_moments_std, ngl)
  G_z = tf.identity(G_z, name='G_z')

  G_z_norm_uint8 = f_to_img(G_z_norm)
  G_z_norm_uint8 = tf.identity(G_z_norm_uint8, name='G_z_norm_uint8')

  # Flatten batch
  nch = int(G_z.get_shape()[-1])
  G_z_padded = tf.pad(G_z, [[0, 0], [0, flat_pad], [0, 0]])
  G_z_flat = tf.reshape(G_z_padded, [-1, nch], name='G_z_flat')

  # Encode to int16
  def float_to_int16(x, name=None):
    x_int16 = x * 32767.
    x_int16 = tf.clip_by_value(x_int16, -32767., 32767.)
    x_int16 = tf.cast(x_int16, tf.int16, name=name)
    return x_int16
  G_z_int16 = float_to_int16(G_z, name='G_z_int16')
  G_z_flat_int16 = float_to_int16(G_z_flat, name='G_z_flat_int16')

  # Create saver
  G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
  global_step = tf.train.get_or_create_global_step()
  saver = tf.train.Saver(G_vars + [global_step])

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


"""
  Generates a preview audio file every time a checkpoint is saved
"""
def preview(args):
  from scipy.io.wavfile import write as wavwrite
  from scipy.signal import freqz

  preview_dir = os.path.join(args.train_dir, 'preview')
  if not os.path.isdir(preview_dir):
    os.makedirs(preview_dir)

  # Load graph
  infer_metagraph_fp = os.path.join(args.train_dir, 'infer', 'infer.meta')
  graph = tf.get_default_graph()
  saver = tf.train.import_meta_graph(infer_metagraph_fp)

  # Generate or restore z_i and z_o
  z_fp = os.path.join(preview_dir, 'z.pkl')
  if os.path.exists(z_fp):
    with open(z_fp, 'rb') as f:
      _zs = pickle.load(f)
  else:
    # Sample z
    samp_feeds = {}
    samp_feeds[graph.get_tensor_by_name('samp_z_n:0')] = args.preview_n
    samp_fetches = {}
    samp_fetches['zs'] = graph.get_tensor_by_name('samp_z:0')
    with tf.Session() as sess:
      _samp_fetches = sess.run(samp_fetches, samp_feeds)
    _zs = _samp_fetches['zs']

    # Save z
    with open(z_fp, 'wb') as f:
      pickle.dump(_zs, f)

  # Set up graph for generating preview images
  feeds = {}
  feeds[graph.get_tensor_by_name('z:0')] = _zs
  feeds[graph.get_tensor_by_name('ngl:0')] = args.specgan_ngl
  feeds[graph.get_tensor_by_name('flat_pad:0')] = _SLICE_LEN // 2
  fetches =  {}
  fetches['step'] = tf.train.get_or_create_global_step()
  fetches['G_z'] = graph.get_tensor_by_name('G_z:0')
  fetches['G_z_flat_int16'] = graph.get_tensor_by_name('G_z_flat_int16:0')

  # Summarize
  G_z = graph.get_tensor_by_name('G_z_flat:0')
  summaries = [
      tf.summary.audio('preview', tf.expand_dims(G_z, axis=0), args.data_sample_rate, max_outputs=1)
  ]
  fetches['summaries'] = tf.summary.merge(summaries)
  summary_writer = tf.summary.FileWriter(preview_dir)

  # Loop, waiting for checkpoints
  ckpt_fp = None
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      print('Preview: {}'.format(latest_ckpt_fp))

      with tf.Session() as sess:
        saver.restore(sess, latest_ckpt_fp)

        _fetches = sess.run(fetches, feeds)

        _step = _fetches['step']

      preview_fp = os.path.join(preview_dir, '{}.wav'.format(str(_step).zfill(8)))
      wavwrite(preview_fp, args.data_sample_rate, _fetches['G_z_flat_int16'])

      summary_writer.add_summary(_fetches['summaries'], _step)

      print('Done')

      ckpt_fp = latest_ckpt_fp

    time.sleep(1)


"""
  Computes inception score every time a checkpoint is saved
"""
def incept(args):
  incept_dir = os.path.join(args.train_dir, 'incept')
  if not os.path.isdir(incept_dir):
    os.makedirs(incept_dir)

  # Load GAN graph
  gan_graph = tf.Graph()
  with gan_graph.as_default():
    infer_metagraph_fp = os.path.join(args.train_dir, 'infer', 'infer.meta')
    gan_saver = tf.train.import_meta_graph(infer_metagraph_fp)
    score_saver = tf.train.Saver(max_to_keep=1)
  gan_z = gan_graph.get_tensor_by_name('z:0')
  gan_ngl = gan_graph.get_tensor_by_name('ngl:0')
  gan_G_z = gan_graph.get_tensor_by_name('G_z:0')[:, :, 0]
  gan_step = gan_graph.get_tensor_by_name('global_step:0')

  # Load or generate latents
  z_fp = os.path.join(incept_dir, 'z.pkl')
  if os.path.exists(z_fp):
    with open(z_fp, 'rb') as f:
      _zs = pickle.load(f)
  else:
    gan_samp_z_n = gan_graph.get_tensor_by_name('samp_z_n:0')
    gan_samp_z = gan_graph.get_tensor_by_name('samp_z:0')
    with tf.Session(graph=gan_graph) as sess:
      _zs = sess.run(gan_samp_z, {gan_samp_z_n: args.incept_n})
    with open(z_fp, 'wb') as f:
      pickle.dump(_zs, f)

  # Load classifier graph
  incept_graph = tf.Graph()
  with incept_graph.as_default():
    incept_saver = tf.train.import_meta_graph(args.incept_metagraph_fp)
  incept_x = incept_graph.get_tensor_by_name('x:0')
  incept_preds = incept_graph.get_tensor_by_name('scores:0')
  incept_sess = tf.Session(graph=incept_graph)
  incept_saver.restore(incept_sess, args.incept_ckpt_fp)

  # Create summaries
  summary_graph = tf.Graph()
  with summary_graph.as_default():
    incept_mean = tf.placeholder(tf.float32, [])
    incept_std = tf.placeholder(tf.float32, [])
    summaries = [
        tf.summary.scalar('incept_mean', incept_mean),
        tf.summary.scalar('incept_std', incept_std)
    ]
    summaries = tf.summary.merge(summaries)
  summary_writer = tf.summary.FileWriter(incept_dir)

  # Loop, waiting for checkpoints
  ckpt_fp = None
  _best_score = 0.
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      print('Incept: {}'.format(latest_ckpt_fp))

      sess = tf.Session(graph=gan_graph)

      gan_saver.restore(sess, latest_ckpt_fp)

      _step = sess.run(gan_step)

      _G_zs = []
      for i in xrange(0, args.incept_n, 100):
        _G_zs.append(sess.run(gan_G_z, {gan_z: _zs[i:i+100], gan_ngl: args.specgan_ngl}))
      _G_zs = np.concatenate(_G_zs, axis=0)

      _preds = []
      for i in xrange(0, args.incept_n, 100):
        _preds.append(incept_sess.run(incept_preds, {incept_x: _G_zs[i:i+100]}))
      _preds = np.concatenate(_preds, axis=0)

      # Split into k groups
      _incept_scores = []
      split_size = args.incept_n // args.incept_k
      for i in xrange(args.incept_k):
        _split = _preds[i * split_size:(i + 1) * split_size]
        _kl = _split * (np.log(_split) - np.log(np.expand_dims(np.mean(_split, 0), 0)))
        _kl = np.mean(np.sum(_kl, 1))
        _incept_scores.append(np.exp(_kl))

      _incept_mean, _incept_std = np.mean(_incept_scores), np.std(_incept_scores)

      # Summarize
      with tf.Session(graph=summary_graph) as summary_sess:
        _summaries = summary_sess.run(summaries, {incept_mean: _incept_mean, incept_std: _incept_std})
      summary_writer.add_summary(_summaries, _step)

      # Save
      if _incept_mean > _best_score:
        score_saver.save(sess, os.path.join(incept_dir, 'best_score'), _step)
        _best_score = _incept_mean

      sess.close()

      print('Done')

      ckpt_fp = latest_ckpt_fp

    time.sleep(1)

  incept_sess.close()


"""
  Calculates and saves dataset moments
"""
def moments(fps, args):
  with tf.name_scope('loader'):
    x_wav = loader.decode_extract_and_batch(
        fps,
        batch_size=1,
        slice_len=_SLICE_LEN,
        decode_fs=args.data_sample_rate,
        decode_num_channels=1,
        decode_fast_wav=args.data_fast_wav,
        decode_parallel_calls=4,
        slice_randomize_offset=False if args.data_first_slice else True,
        slice_first_only=args.data_first_slice,
        slice_overlap_ratio=0. if args.data_first_slice else args.data_overlap_ratio,
        slice_pad_end=True if args.data_first_slice else args.data_pad_end,
        repeat=False,
        shuffle=False,
        shuffle_buffer_size=0,
        prefetch_size=4,
        prefetch_gpu_num=args.data_prefetch_gpu_num)[0, :, 0, 0]

  X = tf.contrib.signal.stft(x_wav, 256, 128, pad_end=True)
  X_mag = tf.abs(X)
  X_lmag = tf.log(X_mag + _LOG_EPS)

  _X_lmags = []
  with tf.Session() as sess:
    while True:
      try:
        _X_lmag = sess.run(X_lmag)
      except:
        break

      _X_lmags.append(_X_lmag)

  _X_lmags = np.concatenate(_X_lmags, axis=0)
  mean, std = np.mean(_X_lmags, axis=0), np.std(_X_lmags, axis=0)

  with open(args.data_moments_fp, 'wb') as f:
    pickle.dump((mean, std), f)


if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'moments', 'preview', 'incept', 'infer'])
  parser.add_argument('train_dir', type=str,
      help='Training directory')

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_dir', type=str,
      help='Data directory')
  data_args.add_argument('--data_moments_fp', type=str,
      help='Dataset moments')
  data_args.add_argument('--data_sample_rate', type=int,
      help='Number of audio samples per second')
  data_args.add_argument('--data_overlap_ratio', type=float,
      help='Overlap ratio [0, 1) between slices')
  data_args.add_argument('--data_first_slice', action='store_true', dest='data_first_slice',
      help='If set, only use the first slice each audio example')
  data_args.add_argument('--data_pad_end', action='store_true', dest='data_pad_end',
      help='If set, use zero-padded partial slices from the end of each audio file')
  data_args.add_argument('--data_normalize', action='store_true', dest='data_normalize',
      help='If set, normalize the training examples')
  data_args.add_argument('--data_fast_wav', action='store_true', dest='data_fast_wav',
      help='If your data is comprised of standard WAV files (16-bit signed PCM or 32-bit float), use this flag to decode audio using scipy (faster) instead of librosa')
  data_args.add_argument('--data_prefetch_gpu_num', type=int,
      help='If nonnegative, prefetch examples to this GPU (Tensorflow device num)')

  specgan_args = parser.add_argument_group('SpecGAN')
  specgan_args.add_argument('--specgan_latent_dim', type=int,
      help='Number of dimensions of the latent space')
  specgan_args.add_argument('--specgan_kernel_len', type=int,
      help='Length of square 2D filter kernels')
  specgan_args.add_argument('--specgan_dim', type=int,
      help='Dimensionality multiplier for model of G and D')
  specgan_args.add_argument('--specgan_batchnorm', action='store_true', dest='specgan_batchnorm',
      help='Enable batchnorm')
  specgan_args.add_argument('--specgan_disc_nupdates', type=int,
      help='Number of discriminator updates per generator update')
  specgan_args.add_argument('--specgan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],
      help='Which GAN loss to use')
  specgan_args.add_argument('--specgan_genr_upsample', type=str, choices=['zeros', 'nn', 'lin', 'cub'],
      help='Generator upsample strategy')
  specgan_args.add_argument('--specgan_ngl', type=int,
      help='Number of Griffin-Lim iterations')

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_batch_size', type=int,
      help='Batch size')
  train_args.add_argument('--train_save_secs', type=int,
      help='How often to save model')
  train_args.add_argument('--train_summary_secs', type=int,
      help='How often to report summaries')

  preview_args = parser.add_argument_group('Preview')
  preview_args.add_argument('--preview_n', type=int,
      help='Number of samples to preview')

  incept_args = parser.add_argument_group('Incept')
  incept_args.add_argument('--incept_metagraph_fp', type=str,
      help='Inference model for inception score')
  incept_args.add_argument('--incept_ckpt_fp', type=str,
      help='Checkpoint for inference model')
  incept_args.add_argument('--incept_n', type=int,
      help='Number of generated examples to test')
  incept_args.add_argument('--incept_k', type=int,
      help='Number of groups to test')

  parser.set_defaults(
    data_dir=None,
    data_moments_fp=None,
    data_sample_rate=16000,
    data_overlap_ratio=0.,
    data_first_slice=False,
    data_pad_end=False,
    data_normalize=False,
    data_fast_wav=False,
    data_prefetch_gpu_num=0,
    specgan_latent_dim=100,
    specgan_kernel_len=5,
    specgan_dim=64,
    specgan_batchnorm=False,
    specgan_disc_nupdates=5,
    specgan_loss='wgan-gp',
    specgan_genr_upsample='zeros',
    specgan_ngl=16,
    train_batch_size=64,
    train_save_secs=300,
    train_summary_secs=120,
    preview_n=32,
    incept_metagraph_fp='./eval/inception/infer.meta',
    incept_ckpt_fp='./eval/inception/best_acc-103005',
    incept_n=5000,
    incept_k=10)

  args = parser.parse_args()

  # Make train dir
  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  # Save args
  with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

  # Load moments
  if args.mode != 'moments' and args.data_moments_fp is not None:
    with open(args.data_moments_fp, 'rb') as f:
      _mean, _std = pickle.load(f)
    setattr(args, 'data_moments_mean', _mean)
    setattr(args, 'data_moments_std', _std)

  # Make model kwarg dicts
  setattr(args, 'specgan_g_kwargs', {
      'kernel_len': args.specgan_kernel_len,
      'dim': args.specgan_dim,
      'use_batchnorm': args.specgan_batchnorm,
      'upsample': args.specgan_genr_upsample
  })
  setattr(args, 'specgan_d_kwargs', {
      'kernel_len': args.specgan_kernel_len,
      'dim': args.specgan_dim,
      'use_batchnorm': args.specgan_batchnorm
  })

  if args.mode == 'train':
    fps = glob.glob(os.path.join(args.data_dir, '*'))
    if len(fps) == 0:
      raise Exception('Did not find any audio files in specified directory')
    print('Found {} audio files in specified directory'.format(len(fps)))
    infer(args)
    train(fps, args)
  elif args.mode == 'moments':
    fps = glob.glob(os.path.join(args.data_dir, '*'))
    print('Found {} audio files in specified directory'.format(len(fps)))
    moments(fps, args)
  elif args.mode == 'preview':
    preview(args)
  elif args.mode == 'incept':
    incept(args)
  elif args.mode == 'infer':
    infer(args)
  else:
    raise NotImplementedError()
