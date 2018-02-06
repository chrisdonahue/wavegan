# WaveGAN

<img src="static/results.png"/>

TensorFlow implementation of WaveGAN (Redacted et al).

## About

<img src="static/wavegan.png"/>

WaveGAN is a GAN approach designed for operation on raw, time-domain audio samples. It is analogous to the DCGAN approach (Radford et al. 2016). The model uses one-dimensional transposed convolutions with longer filters and larger stride than DCGAN, as shown in the figure above.

## Usage

### Requirements

```
pip install tensorflow-gpu==1.4.0
pip install scipy
pip install matplotlib
```

### Build datasets

You can download the datasets from our paper ...

- [Speech Commands Zero through Nine (SC09)](Redacted for blind submission)
- [Drums](Redacted for blind submission)
- [Piano](Redacted for blind submission)

or build your own from any type of audio files

```
python data/make_tfrecord \
	/my/audio/folder/trainset \
	./data/customdataset \
	--ext mp3 \
	--fs 16000 \
	--nshards 64 \
	--slice_len 1.5 \
```

### Train a model

To begin (or resume) training

```
python train_wavegan.py train ./train --data_dir ./data/customdataset \
	--wavegan_genr_pp \
	--wavegan_disc_phaseshuffle 2
```

If your results sound muffled or otherwise strange, try removing `--wavegan_genr_pp` or setting `--wavegan_disc_phaseshuffle 0`. 

To run a script that will dump a preview of fixed latent vectors at each checkpoint on the CPU

```
export CUDA_VISIBLE_DEVICES="-1"
python train_wavegan.py preview ./train
```

To run a (slow) script that will calculate inception score for the SC09 dataset at each checkpoint

```
export CUDA_VISIBLE_DEVICES="-1"
python train_wavegan.py incept ./train
```

## Attribution

Redacted for blind review
