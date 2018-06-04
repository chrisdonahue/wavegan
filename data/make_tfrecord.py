if __name__  == '__main__':
  import argparse
  import glob
  import os
  import random
  import sys

  import numpy as np
  import tensorflow as tf
  from tqdm import tqdm

  parser = argparse.ArgumentParser()

  parser.add_argument('in_dir', type=str)
  parser.add_argument('out_dir', type=str)
  parser.add_argument('--name', type=str)
  parser.add_argument('--ext', type=str)
  parser.add_argument('--fs', type=int)
  parser.add_argument('--nshards', type=int)
  parser.add_argument('--slice_len', type=float)
  parser.add_argument('--first_only', action='store_true', dest='first_only')
  parser.add_argument('--nrg_top_k', action='store_true', dest='nrg_top_k')
  parser.add_argument('--nrg_one_every', type=float)
  parser.add_argument('--nrg_min_per', type=int)
  parser.add_argument('--nrg_max_per', type=int)
  parser.add_argument('--labels', action='store_true', dest='labels')
  parser.add_argument('--labels_whitelist', type=str)

  parser.set_defaults(
      name='examples',
      ext='wav',
      fs=16000,
      nshards=1,
      slice_len=None,
      first_only=False,
      nrg_top_k=False,
      nrg_one_every=5.,
      nrg_min_per=1,
      nrg_max_per=4,
      labels=False,
      labels_whitelist=None)

  args = parser.parse_args()

  labels_whitelist = None
  if args.labels_whitelist is not None:
    labels_whitelist = set([l.strip() for l in args.labels_whitelist.split(',')])

  audio_fps = glob.glob(os.path.join(args.in_dir, '*.{}'.format(args.ext)))
  random.shuffle(audio_fps)

  if args.nshards > 1:
    npershard = max(int(len(audio_fps) // (args.nshards - 1)), 1)
  else:
    npershard = len(audio_fps)

  slice_len_samps = None
  if args.slice_len is not None:
    slice_len_samps = int(args.slice_len * args.fs)

  audio_fp = tf.placeholder(tf.string, [])
  audio_bin = tf.read_file(audio_fp)
  samps = tf.contrib.ffmpeg.decode_audio(audio_bin, args.ext, args.fs, 1)[:, 0]
  if slice_len_samps is not None:
    if args.first_only:
      pad_end = True
    else:
      pad_end = False

    slices = tf.contrib.signal.frame(samps, slice_len_samps, slice_len_samps, axis=0, pad_end=pad_end)

    if args.nrg_top_k:
      nsecs = tf.cast(tf.shape(samps)[0], tf.float32) / args.fs
      k = tf.cast(nsecs / args.nrg_one_every, tf.int32)
      k = tf.maximum(k, args.nrg_min_per)
      k = tf.minimum(k, args.nrg_max_per)

      nrgs = tf.reduce_mean(tf.square(slices), axis=1)
      _, top_k = tf.nn.top_k(nrgs, k)

      slices = tf.gather(slices, top_k, axis=0)

    if args.first_only:
      slices = slices[:1]
  else:
    slices = tf.expand_dims(samps, axis=0)

  sess = tf.Session()

  for i, start_idx in tqdm(enumerate(range(0, len(audio_fps), npershard))):
    shard_name = '{}-{}-of-{}.tfrecord'.format(args.name, str(i).zfill(len(str(args.nshards))), args.nshards)
    shard_fp = os.path.join(args.out_dir, shard_name)

    if not os.path.isdir(args.out_dir):
      os.makedirs(args.out_dir)

    writer = tf.python_io.TFRecordWriter(shard_fp)

    for _audio_fp in audio_fps[start_idx:start_idx+npershard]:
      audio_name = os.path.splitext(os.path.split(_audio_fp)[1])[0]
      if args.labels:
        audio_label, audio_id = audio_name.split('_', 1)

        if labels_whitelist is not None:
          if audio_label not in labels_whitelist:
            continue
      else:
        audio_id = audio_name
        audio_label = ''

      try:
        _slices = sess.run(slices, {audio_fp: _audio_fp})
      except:
        continue

      if _slices.shape[0] == 0 or _slices.shape[1] == 0:
        continue

      for j, _slice in enumerate(_slices):
        example = tf.train.Example(features=tf.train.Features(feature={
          'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=audio_id)),
          'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=audio_label)),
          'slice': tf.train.Feature(int64_list=tf.train.Int64List(value=[j])),
          'samples': tf.train.Feature(float_list=tf.train.FloatList(value=_slice))
        }))

        writer.write(example.SerializeToString())

    writer.close()

  sess.close()
