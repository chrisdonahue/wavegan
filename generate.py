import argparse
import glob
import sys
import os

import librosa
import numpy as np
import tensorflow as tf


def get_arguments():
  parser = argparse.ArgumentParser(description='WaveGan generation script')
  parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from e.g. "(fullpath)/model.ckpt-XXX"')
  parser.add_argument('--train_dir', type=str, help='Training directory')
  parser.add_argument('--wav_out_path', type=str, help='Path to output wav file')
  arguments = parser.parse_args()

  return arguments


def main():
  args = get_arguments()
  infer_dir = os.path.join(args.train_dir, 'infer')
  infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
  tf.reset_default_graph()
  saver = tf.train.import_meta_graph(infer_metagraph_fp)
  graph = tf.get_default_graph()
  sess = tf.InteractiveSession()
  saver.restore(sess, args.checkpoint)
  _z = (np.random.rand(1, 100) * 2.) - 1.
  z = graph.get_tensor_by_name('z:0')
  G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
  waveform = sess.run(G_z, {z: _z})
  librosa.output.write_wav(args.wav_out_path, waveform[0, :], 16000)
  sess.close()

  print('Finished generating.')


if __name__ == '__main__':
  main()
