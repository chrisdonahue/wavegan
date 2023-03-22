from __future__ import print_function
import numpy as np
import tensorflow as tf
from scipy import linalg

from six.moves import xrange


# this function is taken from 
# https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean



def inception_score(
        audio_fps_real,
        audio_fps_fake,
        k,
        metagraph_fp,
        ckpt_fp,
        batch_size=100,
        tf_ffmpeg_ext=None,
        fix_length=False):
    # Compute IS and FID scores
    use_tf_ffmpeg = tf_ffmpeg_ext is not None
    if not use_tf_ffmpeg:
        from scipy.io.wavfile import read as wavread

    if len(audio_fps_real) % k != 0:
        raise Exception('Number of audio files ({}) is not divisible by k ({})'.format(len(audio_fps_real), k))
    group_size = len(audio_fps_real) // k

    assert(len(audio_fps_real)==len(audio_fps_fake))

    # Restore graph
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(metagraph_fp)
        
        if use_tf_ffmpeg:
            x_fp = tf.placeholder(tf.string, [])
            x_bin = tf.read_file(x_fp)
            x_samps = tf.contrib.ffmpeg.decode_audio(x_bin, tf_ffmpeg_ext, 16000, 1)[:, 0]
    x = graph.get_tensor_by_name('x:0')
    activations = graph.get_tensor_by_name('classifier/dense/BiasAdd:0')
    scores = graph.get_tensor_by_name('scores:0')

    # Restore weights
    sess = tf.Session(graph=graph)
    saver.restore(sess, ckpt_fp)

    # Evaluate audio
    _all_scores_fake = []
    _all_scores_real = []
    _all_activations_real = []
    _all_activations_fake = []
    for i in xrange(0, len(audio_fps_real), batch_size):
        batch_real = audio_fps_real[i:i+batch_size]
        batch_fake = audio_fps_fake[i:i+batch_size]

        # Load audio files
        _xs = []
        for audio_fp in batch_real:
            if use_tf_ffmpeg:
                _x = sess.run(x_samps, {x_fp: audio_fp})
            else:
                fs, _x = wavread(audio_fp)
                if fs != 16000:
                    raise Exception('Invalid sample rate ({})'.format(fs))
            if _x.dtype==np.int16:
                _x = _x.astype(np.float32)
                _x /= 32767.

            if _x.ndim != 1:
                raise Exception('Invalid shape ({})'.format(_x.shape))

            if fix_length:
                _x = _x[:16384]
                #_x = _x[-16384:]
                _x = np.pad(_x, (0, 16384 - _x.shape[0]), 'constant')

            if _x.shape[0] != 16384:
                raise Exception('Invalid number of samples ({})'.format(_x.shape[0]))

            _xs.append(_x)

        # Compute model scores
        _scores, _activations = sess.run([scores, activations], {x: _xs})
        _all_scores_real.append(_scores)
        _all_activations_real.append(_activations)

        # Load audio files
        _xs = []
        for audio_fp in batch_fake:
            if use_tf_ffmpeg:
                _x = sess.run(x_samps, {x_fp: audio_fp})
            else:
                fs, _x = wavread(audio_fp)
                if fs != 16000:
                    raise Exception('Invalid sample rate ({})'.format(fs))
            if _x.dtype==np.int16:
                _x = _x.astype(np.float32)
                _x /= 32767.

            if _x.ndim != 1:
                raise Exception('Invalid shape ({})'.format(_x.shape))

            if fix_length:
                _x = _x[:16384]
                #_x = _x[-16384:]
                _x = np.pad(_x, (0, 16384 - _x.shape[0]), 'constant')

            if _x.shape[0] != 16384:
                raise Exception('Invalid number of samples ({})'.format(_x.shape[0]))

            _xs.append(_x)

        # Compute model scores
        _scores, _activations = sess.run([scores, activations], {x: _xs})
        _all_scores_fake.append(_scores)
        _all_activations_fake.append(_activations)

    sess.close()

    # Find labels
    _all_scores_fake = np.concatenate(_all_scores_fake, axis=0)
    _all_scores_real = np.concatenate(_all_scores_real, axis=0)
    _all_activations_fake = np.concatenate(_all_activations_fake, axis=0)
    _all_activations_real = np.concatenate(_all_activations_real, axis=0)
    _all_labels_fake = np.argmax(_all_scores_fake, axis=1)
    _all_labels_real = np.argmax(_all_scores_fake, axis=1)

    # Compute inception scores
    _inception_scores_fake = []
    for i in xrange(k):
        _group = _all_scores_fake[i * group_size:(i + 1) * group_size]
        _kl = _group * (np.log(_group) - np.log(np.expand_dims(np.mean(_group, 0), 0)))
        _kl = np.mean(np.sum(_kl, 1))
        _inception_scores_fake.append(np.exp(_kl))

    _inception_scores_real = []
    for i in xrange(k):
        _group = _all_scores_real[i * group_size:(i + 1) * group_size]
        _kl = _group * (np.log(_group) - np.log(np.expand_dims(np.mean(_group, 0), 0)))
        _kl = np.mean(np.sum(_kl, 1))
        _inception_scores_real.append(np.exp(_kl))

    # Compute the FID scores
    mu_real = np.mean(_all_activations_real, axis=0)
    sigma_real = np.cov(_all_activations_real, rowvar=False)
    mu_fake = np.mean(_all_activations_fake, axis=0)
    sigma_fake = np.cov(_all_activations_fake, rowvar=False)

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    ret = (np.mean(_inception_scores_fake),
           np.std(_inception_scores_fake),
           _all_labels_fake,
           np.mean(_inception_scores_real),
           np.std(_inception_scores_real),
           _all_labels_real,
           fid)

    return ret

if __name__ == '__main__':
    import argparse
    import glob
    import os
    import random
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str,
            help='Directory with 16-bit signed integer PCM WAV files at 16kHz')
    parser.add_argument('--real_audio_dir', type=str,
            help='Directory with 16-bit signed integer PCM WAV files at 16kHz')
    parser.add_argument('--fix_length', action='store_true', dest='fix_length',
            help='If set, pad or trim audio files to length 16384')
    parser.add_argument('--labels_fp', type=str,
            help='If set, write model predictions to this file')
    parser.add_argument('--metagraph_fp', type=str,
            help='MetaGraph for classifier; must have tensors x:0 [None, 16384] and scores:0 [None, 10]')
    parser.add_argument('--ckpt_fp', type=str,
            help='Checkpoint for metagraph')
    parser.add_argument('--n', type=int,
            help='Number of samples to test')
    parser.add_argument('--k', type=int,
            help='Number of subsets to score')
    parser.add_argument('--batch_size', type=int,
            help='Evaluate audio in batches of this size')
    parser.add_argument('--tf_ffmpeg_ext', type=str,
            help='If set, uses ffmpeg to decode audio files with specified extension through tensorflow')

    parser.set_defaults(
        audio_dir=None,
        real_audio_dir=None,
        fix_length=False,
        labels_fp=None,
        metagraph_fp='infer.meta',
        ckpt_fp='best_acc-103005',
        n=50000,
        k=10,
        batch_size=100,
        tf_ffmpeg_ext=None)

    args = parser.parse_args()

    # Find audio files
    if args.audio_dir is None:
        raise Exception('No audio directory specified')
    ext = 'wav' if args.tf_ffmpeg_ext is None else args.tf_ffmpeg_ext
    audio_fps = sorted(glob.glob(os.path.join(args.audio_dir, '*.{}'.format(ext))))
    random.seed(0)
    random.shuffle(audio_fps)
    if len(audio_fps) < args.n:
        raise Exception('Found fewer ({}) than specified ({}) audio files'.format(len(audio_fps), args.n))
    audio_fps = audio_fps[:args.n]

    if args.real_audio_dir is None:
        raise Exception('No real audio directory specified')
    ext = 'wav' if args.tf_ffmpeg_ext is None else args.tf_ffmpeg_ext
    real_audio_fps = sorted(glob.glob(os.path.join(args.real_audio_dir, '*.{}'.format(ext))))
    random.seed(0)
    random.shuffle(real_audio_fps)
    real_audio_fps = real_audio_fps[:args.n]

    # Compute scores
    fake_mean, fake_std, fake_labels, real_mean, real_std, real_labels, fid = inception_score(
            real_audio_fps,
            audio_fps,
            args.k,
            args.metagraph_fp,
            args.ckpt_fp,
            batch_size=args.batch_size,
            tf_ffmpeg_ext=args.tf_ffmpeg_ext,
            fix_length=args.fix_length)
    print('Real inception score: {} +- {}'.format(real_mean, real_std))
    print('Fake inception score: {} +- {}'.format(fake_mean, fake_std))
    print('FID score: {}'.format(fid))

    print('p(y)')
    for i in xrange(10):
        n = len([x for x in fake_labels if x == i])
        print('{}: {}'.format(i, n / float(args.n)))

    # Save labels
    if args.labels_fp is not None:
        labels_txt = []
        for audio_fp, label in zip(audio_fps, labels):
            labels_txt.append(','.join([audio_fp, str(label)]))
        with open(args.labels_fp, 'w') as f:
            f.write('\n'.join(labels_txt))
