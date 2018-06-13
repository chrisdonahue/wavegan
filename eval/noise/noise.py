from __future__ import print_function

if __name__ == '__main__':
  import argparse
  import cPickle as pickle
  import glob
  import os
  import random
  import numpy as np
  from scipy.io.wavfile import read as wavread
  from tqdm import tqdm

  parser = argparse.ArgumentParser()
  parser.add_argument('--noise_env_fp', type=str,
      help='Frequency-domain noise envelope (normalized between 0 and 1)')
  parser.add_argument('--wav_dir', type=str,
      help='Directory to test')
  parser.add_argument('--n', type=int,
      help='Limit to this many examples')

  args = parser.parse_args()

  # Load noise envelope
  with open(args.noise_env_fp, 'rb') as f:
    noise_env = pickle.load(f)

  # Retrieve wav fps
  wav_fps = glob.glob(os.path.join(args.wav_dir, '*.wav'))
  wav_fps = sorted(wav_fps)
  random.seed(0)
  random.shuffle(wav_fps)
  if args.n is not None:
    wav_fps = wav_fps[:args.n]

  # Load wavs
  xs = []
  for wav_fp in tqdm(wav_fps):
    fs, x = wavread(wav_fp)
    assert fs == 16000
    x = x.astype(np.float32)
    x /= 32767.
    xs.append(x)
  x = np.array(xs)

  # Normalize RMS across dataset (not per-wav)
  x_rms = np.sqrt(np.mean(np.square(x.reshape(-1))))
  gain = 1.0 / x_rms
  x *= gain

  # Compute FFT
  X = np.fft.rfft(xs, 16384)
  X_mag = np.abs(X)
  if 'log' in args.noise_env_fp:
    X_mag = np.log(X_mag + 1e-6)
  X_weighted_mean = np.average(X_mag, weights=noise_env, axis=1)

  print('{} +- {}'.format(np.mean(X_weighted_mean), np.std(X_weighted_mean)))
