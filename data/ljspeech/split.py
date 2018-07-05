from __future__ import print_function
from collections import defaultdict
import glob
import os
import random
import sys

from tqdm import tqdm

ljspeech_dir, out_dir = sys.argv[1:3]
dry_run = False

wav_fps = set(glob.glob(os.path.join(ljspeech_dir, 'wavs', '*.wav')))
split_to_wav_fps = defaultdict(list)
for wav_fp in wav_fps:
  split = wav_fp.split('/')[-1].split('-')[0]
  split_to_wav_fps[split].append(wav_fp)
splits = set(split_to_wav_fps.keys())
assert len(splits) == 50

random.seed(0)
train_splits = random.sample(splits, 40)
splits -= set(train_splits)
valid_splits = random.sample(splits, 5)
test_splits = splits - set(valid_splits)

train_fps = sum([split_to_wav_fps[s] for s in train_splits], [])
valid_fps = sum([split_to_wav_fps[s] for s in valid_splits], [])
test_fps = sum([split_to_wav_fps[s] for s in test_splits], [])

for split_name, split in zip(['train', 'valid', 'test'], [train_fps, valid_fps, test_fps]):
  for wav_fp in tqdm(split):
    label, wav_name = wav_fp.split('/')[-1].split('-')
    out_fp = os.path.join(out_dir, split_name, '{}_{}'.format(label, wav_name))

    if dry_run:
      print('-' * 80)
      print(wav_fp)
      print(out_fp)
    else:
      os.rename(wav_fp, out_fp)
