import tensorflow as tf


def conv2d_transpose(
    inputs,
    filters,
    kernel_len,
    stride=2,
    padding='same',
    upsample='zeros'):
  if upsample == 'zeros':
    return tf.layers.conv2d_transpose(
        inputs,
        filters,
        kernel_len,
        strides=(stride, stride),
        padding='same')
  elif upsample in ['nn', 'linear', 'cubic']:
    batch_size = tf.shape(inputs)[0]
    _, h, w, nch = inputs.get_shape().as_list()

    x = inputs

    if upsample == 'nn':
      upsampler = tf.image.resize_nearest_neighbor
    elif upsample == 'linear':
      upsampler = tf.image.resize_bilinear
    else:
      upsampler = tf.image.resize_bicubic

    x = upsampler(x, [h * stride, w * stride])
    
    return tf.layers.conv2d(
        x,
        filters,
        kernel_len,
        strides=(1, 1),
        padding='same')
  else:
    raise NotImplementedError


"""
  Input: [None, 100]
  Output: [None, 128, 128, 1]
"""
def SpecGANGenerator(
    z,
    kernel_len=5,
    dim=64,
    use_batchnorm=False,
    upsample='zeros',
    train=False):
  batch_size = tf.shape(z)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x

  # FC and reshape for convolution
  # [100] -> [4, 4, 1024]
  output = z
  with tf.variable_scope('z_project'):
    output = tf.layers.dense(output, 4 * 4 * dim * 16)
    output = tf.reshape(output, [batch_size, 4, 4, dim * 16])
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 0
  # [4, 4, 1024] -> [8, 8, 512]
  with tf.variable_scope('upconv_0'):
    output = conv2d_transpose(output, dim * 8, kernel_len, 2, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 1
  # [8, 8, 512] -> [16, 16, 256]
  with tf.variable_scope('upconv_1'):
    output = conv2d_transpose(output, dim * 4, kernel_len, 2, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 2
  # [16, 16, 256] -> [32, 32, 128]
  with tf.variable_scope('upconv_2'):
    output = conv2d_transpose(output, dim * 2, kernel_len, 2, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 3
  # [32, 32, 128] -> [64, 64, 64]
  with tf.variable_scope('upconv_3'):
    output = conv2d_transpose(output, dim, kernel_len, 2, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 4
  # [64, 64, 64] -> [128, 128, 1]
  with tf.variable_scope('upconv_4'):
    output = conv2d_transpose(output, 1, kernel_len, 2, upsample=upsample)
  output = tf.nn.tanh(output)

  # Automatically update batchnorm moving averages every time G is used during training
  if train and use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
    assert len(update_ops) == 10
    with tf.control_dependencies(update_ops):
      output = tf.identity(output)

  return output


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


"""
  Input: [None, 128, 128, 1]
  Output: [None] (linear) output
"""
def SpecGANDiscriminator(
    x,
    kernel_len=5,
    dim=64,
    use_batchnorm=False):
  batch_size = tf.shape(x)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  # Layer 0
  # [128, 128, 1] -> [64, 64, 64]
  output = x
  with tf.variable_scope('downconv_0'):
    output = tf.layers.conv2d(output, dim, kernel_len, 2, padding='SAME')
  output = lrelu(output)

  # Layer 1
  # [64, 64, 64] -> [32, 32, 128]
  with tf.variable_scope('downconv_1'):
    output = tf.layers.conv2d(output, dim * 2, kernel_len, 2, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)

  # Layer 2
  # [32, 32, 128] -> [16, 16, 256]
  with tf.variable_scope('downconv_2'):
    output = tf.layers.conv2d(output, dim * 4, kernel_len, 2, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)

  # Layer 3
  # [16, 16, 256] -> [8, 8, 512]
  with tf.variable_scope('downconv_3'):
    output = tf.layers.conv2d(output, dim * 8, kernel_len, 2, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)

  # Layer 4
  # [8, 8, 512] -> [4, 4, 1024]
  with tf.variable_scope('downconv_4'):
    output = tf.layers.conv2d(output, dim * 16, kernel_len, 2, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)

  # Flatten
  output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16])

  # Connect to single logit
  with tf.variable_scope('output'):
    output = tf.layers.dense(output, 1)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
