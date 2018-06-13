from __future__ import print_function
import glob
import os
import sys

from tqdm import tqdm

train_dir, out_dir = sys.argv[1:3]
dry_run = False

with open(os.path.join(train_dir, 'validation_list.txt'), 'r') as f:
  valid_fps = set([os.path.join(train_dir, 'audio', l.strip()) for l in f.read().splitlines()])
with open(os.path.join(train_dir, 'testing_list.txt'), 'r') as f:
  test_fps = set([os.path.join(train_dir, 'audio', l.strip()) for l in f.read().splitlines()])

wav_fps = set(glob.glob(os.path.join(train_dir, 'audio', '*', '*.wav')))

train_fps = wav_fps - valid_fps - test_fps

for split_name, split in zip(['train', 'valid', 'test'], [train_fps, valid_fps, test_fps]):
  for wav_fp in tqdm(split):
    wav_name = os.path.split(wav_fp)[1]
    label = wav_fp.split('/')[-2].title()
    out_fp = os.path.join(out_dir, split_name, '{}_{}'.format(label, wav_name))

    if dry_run:
      print('-' * 80)
      print(wav_fp)
      print(out_fp)
    else:
      os.rename(wav_fp, out_fp)
