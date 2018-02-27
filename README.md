# WaveGAN

<img src="static/results.png"/>
<img src="static/wavegan.png"/>

TensorFlow implementation of WaveGAN (Donahue et al. 2018). WaveGAN is a GAN approach designed for operation on raw, time-domain audio samples. It is related to the DCGAN approach (Radford et al. 2016), a popular GAN model designed for image synthesis. WaveGAN uses one-dimensional transposed convolutions with longer filters and larger stride than DCGAN, as shown in the figure above.

## Usage

### Requirements

```
pip install tensorflow-gpu==1.4.0
pip install scipy
pip install matplotlib
```

### Build datasets

You can download the datasets from our paper ...

- [Speech Commands Zero through Nine (SC09)](https://drive.google.com/open?id=1qRdAWmjfWwfWIu-Qk7u9KQKGINC52ZwB)
- [Drums](https://drive.google.com/open?id=1nKIWosguCSsEzYomHWfWmmu3RlLTMUIE)
- [Piano](https://drive.google.com/open?id=1REGUUFhFcp-L_5LngJp4oZouGNBy8DPh)

or build your own from any type of audio files:

```
python data/make_tfrecord.py \
	/my/audio/folder/trainset \
	./data/customdataset \
	--ext mp3 \
	--fs 16000 \
	--nshards 64 \
	--slice_len 1.5 \
```

### Train WaveGAN

To begin (or resume) training

```
python train_wavegan.py train ./train \
	--data_dir ./data/customdataset
```

If your results are unsatisfactory, try adding a post-processing filter with `--wavegan_genr_pp` or removing phase shuffle with `--wavegan_disc_phaseshuffle 0`. 

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

To back up checkpoints every hour (GAN training will occasionally collapse)

```
python backup.py ./train 60
```

### Train SpecGAN

Compute dataset moments to use for normalization

```
export CUDA_VISIBLE_DEVICES="-1"
python train_specgan.py moments ./train \
	--data_dir ./data/customdataset \
	--data_moments_fp ./train/moments.pkl
```


To begin (or resume) training

```
python train_specgan.py train ./train \
	--data_dir ./data/customdataset \
	--data_moments_fp ./train/moments.pkl
```

To run a script that will dump a preview of fixed latent vectors at each checkpoint on the CPU

```
export CUDA_VISIBLE_DEVICES="-1"
python train_specgan.py preview ./train \
	--data_moments_fp ./train/moments.pkl
```

To run a (slow) script that will calculate inception score for the SC09 dataset at each checkpoint

```
export CUDA_VISIBLE_DEVICES="-1"
python train_specgan.py incept ./train \
	--data_moments_fp ./train/moments.pkl
```

To back up checkpoints every hour (GAN training will occasionally collapse)

```
python backup.py ./train 60
```
