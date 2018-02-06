import tensorflow as tf
from scipy.io.wavfile import read as wavread
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
  import argparse
  import cPickle as pickle
  import glob
  import os
  import random
  import sys

  parser = argparse.ArgumentParser()
  parser.add_argument('--audio_dir', type=str,
      help='Directory with 16-bit signed integer PCM wav files at 16kHz')
  parser.add_argument('--out_fp', type=str,
      help='Output file with audio features')
  parser.add_argument('--n', type=int,
      help='Limit the number of items for comparison')

  parser.set_defaults(
      n=None)

  args = parser.parse_args()

  wav_fps = sorted(glob.glob(os.path.join(args.audio_dir, '*.wav')))
  random.seed(0)
  random.shuffle(wav_fps)
  if args.n is not None:
    wav_fps = wav_fps[:args.n]

  # Graph to calculate feats
  x = tf.placeholder(tf.float32, [None])
  x_trim = x[:16384]
  x_trim = tf.pad(x_trim, [[0, 16384 - tf.shape(x_trim)[0]]])
  X = tf.contrib.signal.stft(x_trim, 2048, 128, pad_end=True)
  X_mag = tf.abs(X)
  W_mel = tf.contrib.signal.linear_to_mel_weight_matrix(
      num_mel_bins=128,
      num_spectrogram_bins=1025,
      sample_rate=16000,
      lower_edge_hertz=40.,
      upper_edge_hertz=7800.,
  )
  X_mel = tf.matmul(X_mag, W_mel)
  X_lmel = tf.log(X_mel + 1e-6)
  X_feat = X_lmel

  # Calculate feats for each wav file
  with tf.Session() as sess:
    _X_feats = []
    for wav_fp in tqdm(wav_fps):
      _, _x = wavread(wav_fp)

      _X_feats.append(sess.run(X_feat, {x: _x}))
    _X_feats = np.array(_X_feats)

  with open(args.out_fp, 'wb') as f:
    pickle.dump(_X_feats, f)
