from __future__ import print_function
import os
import shutil

bundle_dir = 'bundle'

paths = [
  'ckpts/drums',
  'css',
  'img',
  'js',
  'index.html'
]

if os.path.exists(bundle_dir):
  shutil.rmtree(bundle_dir)

for path in paths:
  out_path = os.path.join(bundle_dir, path)
  print('{}->{}'.format(path, out_path))

  if os.path.isdir(path):
    shutil.copytree(path, out_path)
  else:
    out_dir = os.path.split(out_path)[0]
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    shutil.copy(path, out_path)

wavegan_cfg_fp = os.path.join(bundle_dir, 'js', 'wavegan_cfg.js')
with open(wavegan_cfg_fp, 'r') as f:
  wavegan_cfg = f.read()

wavegan_cfg = wavegan_cfg.replace('var debug = true;', 'var debug = false;')

with open(wavegan_cfg_fp, 'w') as f:
  f.write(wavegan_cfg)
