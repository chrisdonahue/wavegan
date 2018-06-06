# WaveGAN

<img src="static/wavegan.png"/>
<img src="static/results.png"/>

Official TensorFlow implementation of WaveGAN (Donahue et al. 2018) ([paper](https://arxiv.org/abs/1802.04208)) ([demo](https://chrisdonahue.github.io/wavegan/)) ([sound examples](http://wavegan-v1.s3-website-us-east-1.amazonaws.com)). WaveGAN is a GAN approach designed for operation on raw, time-domain audio samples. It is related to the DCGAN approach (Radford et al. 2016), a popular GAN model designed for image synthesis. WaveGAN uses one-dimensional transposed convolutions with longer filters and larger stride than DCGAN, as shown in the figure above.

## Usage

### Requirements

```
# Will likely also work with newer versions of Tensorflow
pip install tensorflow-gpu==1.4.0
pip install scipy
pip install matplotlib
```

### Build datasets

You can download the datasets from our paper bundled as TFRecords ...

- [Speech Commands Zero through Nine (SC09)](https://drive.google.com/open?id=1qRdAWmjfWwfWIu-Qk7u9KQKGINC52ZwB) alternate link: [(raw WAV files)](http://deepyeti.ucsd.edu/cdonahue/sc09.tar.gz)
- [Drums](https://drive.google.com/open?id=1nKIWosguCSsEzYomHWfWmmu3RlLTMUIE)
- [Piano](https://drive.google.com/open?id=1REGUUFhFcp-L_5LngJp4oZouGNBy8DPh) alternate link: [(raw WAV files)](http://deepyeti.ucsd.edu/cdonahue/mancini_piano.tar.gz)

or build your own from directories of audio files:

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

### Generation

The training scripts for both WaveGAN and SpecGAN create simple TensorFlow MetaGraphs for generating audio waveforms, located in the training directory. A simple usage is below; see [this Colab notebook](https://colab.research.google.com/drive/1e9o2NB2GDDjadptGr3rwQwTcw-IrFOnm) for additional features.

```py
import tensorflow as tf
from IPython.display import display, Audio

# Load the graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph('infer.meta')
graph = tf.get_default_graph()
sess = tf.InteractiveSession()
saver.restore(sess, 'model.ckpt')

# Create 50 random latent vectors z
_z = (np.random.rand(50, 100) * 2.) - 1

# Synthesize G(z)
z = graph.get_tensor_by_name('z:0')
G_z = graph.get_tensor_by_name('G_z:0')
_G_z = sess.run(G_z, {z: _z})

# Play audio in notebook
display(Audio(_G_z[0], rate=16000))
```

### Evaluation

Our [paper](https://arxiv.org/abs/1802.04208) uses Inception score to (roughly) measure model performance. If you would like to compare to our reported numbers directly, you may run [this script](https://github.com/chrisdonahue/wavegan/blob/master/eval/inception/score.py) on a directory of 50,000 WAV files with 16384 samples each.

```
python score.py --audio_dir wavs
```


To reproduce our paper results (9.18 +- 0.04) for the SC09 ([download](http://deepyeti.ucsd.edu/cdonahue/sc09.tar.gz)) training dataset, run

```
python score.py --audio_dir sc09/train  --fix_length --n 18620
```



### Attribution

If you use this code in your research, cite via the following BibTeX:

```
@article{donahue2018wavegan,
  title={Synthesizing Audio with Generative Adversarial Networks},
  author={Donahue, Chris and McAuley, Julian and Puckette, Miller},
  journal={arXiv:1802.04208},
  year={2018}
}
```
