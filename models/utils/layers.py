import tensorflow as tf


# from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
def conv(batch_input,
         out_channels,
         filter_size=3,
         stride=1,
         pad_mode="CONSTANT",
         padding="VALID",
         name="conv"):
  with tf.compat.v1.variable_scope(name):
    in_channels = batch_input.get_shape()[3]
    kernel = tf.compat.v1.get_variable(
        "kernel", [filter_size, filter_size, in_channels, out_channels],
        dtype=tf.compat.v1.float32,
        initializer=tf.compat.v1.random_normal_initializer(0, 0.02))
    padded_input = tf.compat.v1.pad(
        batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    conv = tf.compat.v1.nn.conv2d(
        padded_input, kernel, [1, stride, stride, 1], padding=padding)
    return conv


# from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
def deconv(batch_input, out_channels):
  with tf.compat.v1.variable_scope("deconv"):
    batch, in_height, in_width, in_channels = [
        int(d) for d in batch_input.get_shape()
    ]
    kernel = tf.compat.v1.get_variable(
        "kernel", [4, 4, out_channels, in_channels],
        dtype=tf.compat.v1.float32,
        initializer=tf.compat.v1.random_normal_initializer(0, 0.02))
    conv = tf.compat.v1.nn.conv2d_transpose(
        batch_input,
        kernel, [batch, in_height * 2, in_width * 2, out_channels],
        [1, 2, 2, 1],
        padding="SAME")
    return conv


# from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
def batch_norm(x):
  with tf.compat.v1.variable_scope("batchnorm"):
    x = tf.compat.v1.identity(x)

    channels = x.get_shape()[3]
    offset = tf.compat.v1.get_variable(
        "offset", [channels],
        dtype=tf.compat.v1.float32,
        initializer=tf.compat.v1.zeros_initializer())
    scale = tf.compat.v1.get_variable(
        "scale", [channels],
        dtype=tf.compat.v1.float32,
        initializer=tf.compat.v1.random_normal_initializer(1.0, 0.02))
    variance_epsilon = 1e-5
    normalized, _, _ = tf.compat.v1.nn.fused_batch_norm(
        x, scale, offset, epsilon=variance_epsilon)
    return normalized


# from https://github.com/hardikbansal/CycleGAN/blob/master/layers.py
def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

  with tf.compat.v1.variable_scope(name):
    if alt_relu_impl:
      f1 = 0.5 * (1 + leak)
      f2 = 0.5 * (1 - leak)

      return f1 * x + f2 * abs(x)
    else:
      return tf.compat.v1.maximum(x, leak * x)


# from https://github.com/hardikbansal/CycleGAN/blob/master/layers.py
def instance_norm(x):

  with tf.compat.v1.variable_scope("instance_norm"):
    epsilon = 1e-5
    mean, var = tf.compat.v1.nn.moments(x, [1, 2], keep_dims=True)
    scale = tf.compat.v1.get_variable(
        'scale', [x.get_shape()[-1]],
        initializer=tf.compat.v1.truncated_normal_initializer(mean=1.0, stddev=0.02))
    offset = tf.compat.v1.get_variable(
        'offset', [x.get_shape()[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
    out = scale * tf.compat.v1.div(x - mean, tf.compat.v1.sqrt(var + epsilon)) + offset

    return out


def general_conv1d(*args, **kwargs):
  return general_conv(tf.compat.v1.layers.conv1d, *args, **kwargs)


def general_conv2d(*args, **kwargs):
  return general_conv(tf.compat.v1.layers.conv2d, *args, **kwargs)


# modified from https://github.com/hardikbansal/CycleGAN/blob/master/layers.py
def general_conv(conv_layer,
                 x,
                 num_filters=64,
                 filter_size=7,
                 stride=1,
                 stddev=0.02,
                 padding="VALID",
                 name="conv",
                 do_norm="instance",
                 do_relu=True,
                 relufactor=0):
  with tf.compat.v1.variable_scope(name):
    conv = conv_layer(
        x,
        num_filters,
        filter_size,
        stride,
        padding,
        activation=None,
        kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev),
        bias_initializer=tf.compat.v1.constant_initializer(0.0))
    if do_norm == "layer":
      conv = tf.compat.v1.contrib.layers.layer_norm(conv)
    elif do_norm == "instance":
      conv = instance_norm(conv)

    if do_relu:
      if (relufactor == 0):
        conv = tf.compat.v1.nn.relu(conv, "relu")
      else:
        conv = lrelu(conv, relufactor, "lrelu")

    return conv


# modified from https://github.com/hardikbansal/CycleGAN/blob/master/layers.py
def general_deconv2d(x,
                     outshape,
                     num_filters=64,
                     filter_size=7,
                     stride=1,
                     stddev=0.02,
                     padding="VALID",
                     name="deconv2d",
                     do_norm=True,
                     do_relu=True,
                     relufactor=0):
  with tf.compat.v1.variable_scope(name):

    conv = tf.compat.v1.contrib.layers.conv2d_transpose(
        x,
        num_filters, [filter_size, filter_size], [stride, stride],
        padding,
        activation_fn=None,
        weights_initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev),
        biases_initializer=tf.compat.v1.constant_initializer(0.0))

    if do_norm:
      conv = instance_norm(conv)

    if do_relu:
      if (relufactor == 0):
        conv = tf.compat.v1.nn.relu(conv, "relu")
      else:
        conv = lrelu(conv, relufactor, "lrelu")

    return conv


# modified from https://github.com/hardikbansal/CycleGAN/blob/master/model.py
def build_resnet_block(conv_layer,
                       x,
                       dim,
                       filter_size=3,
                       pad=True,
                       name="resnet"):

  with tf.compat.v1.variable_scope(name):
    if pad:
      out_res = tf.compat.v1.pad(x, [[0, 0]] + [[filter_size // 2, filter_size // 2]] *
                       (len(x.shape) - 2) + [[0, 0]], "REFLECT")
    else:
      out_res = x
    out_res = conv_layer(
        x=out_res,
        num_filters=dim,
        filter_size=filter_size,
        stride=1,
        stddev=0.02,
        padding="VALID",
        name="c1")
    if pad:
      out_res = tf.compat.v1.pad(out_res,
                       [[0, 0]] + [[filter_size // 2, filter_size // 2]] *
                       (len(x.shape) - 2) + [[0, 0]], "REFLECT")
    out_res = conv_layer(
        x=out_res,
        num_filters=dim,
        filter_size=filter_size,
        stride=1,
        stddev=0.02,
        padding="VALID",
        name="c2",
        do_relu=False)

    return tf.compat.v1.nn.relu(out_res + x)


def build_n_layer_conv_stack(conv_layer,
                             x,
                             filter_size,
                             num_discrim_filters,
                             n=3,
                             do_norm="instance"):
  """Standard convolutional stack from CycleGAN paper."""
  x = conv_layer(
      x=x,
      num_filters=num_discrim_filters,
      filter_size=filter_size,
      stride=2,
      stddev=0.02,
      padding="SAME",
      name="c1",
      do_norm=False,
      relufactor=0.2)
  for i in range(n):
    x = conv_layer(
        x=x,
        num_filters=num_discrim_filters * 2**(i + 1),
        filter_size=filter_size,
        stride=2,
        stddev=0.02,
        padding="SAME",
        name="c%d" % (i + 2),
        do_norm=do_norm,
        relufactor=0.2)
  x = conv_layer(
      x=x,
      num_filters=1,
      filter_size=filter_size,
      stride=1,
      stddev=0.02,
      padding="SAME",
      name="c%d" % (n + 2),
      do_norm=False,
      do_relu=False)
  return x


def timing(x, params):
  if params.timing_type == "transformer":
    shape = tf.compat.v1.shape(x)
    num = tf.compat.v1.reshape(tf.compat.v1.range(tf.compat.v1.to_float(shape[1])), [1, -1, 1])
    num = tf.compat.v1.tile(num, [shape[0], 1, shape[2]])
    denom = tf.compat.v1.reshape(tf.compat.v1.range(tf.compat.v1.to_float(shape[2])), [1, 1, -1])
    denom = tf.compat.v1.tile(denom, [shape[0], shape[1], 1])
    denom = 10000**(denom / tf.compat.v1.to_float(shape[2]))

    sine_timing = tf.compat.v1.sin(num / denom)
    cos_timing = tf.compat.v1.cos(num / denom)
    layerings = tf.compat.v1.tile([[[True, False]]], [shape[0], shape[1], shape[2] // 2])
    timing = tf.compat.v1.where(layerings, sine_timing, cos_timing)

    return x + timing
  elif params.timing_type == "concat":
    timing = tf.compat.v1.get_variable(
        "timing",
        shape=[params.sample_length, params.hidden_size],
        dtype=tf.compat.v1.float32,
        initializer=tf.compat.v1.random_normal_initializer(
            mean=0.0, stddev=1.0))
    timing = tf.compat.v1.tile(tf.compat.v1.expand_dims(timing, 0), [params.batch_size, 1, 1])
    timing = timing[:tf.compat.v1.shape(x)[0], :tf.compat.v1.shape(x)[1], :]
    return tf.compat.v1.concat([x, timing], 2)
  else:
    raise Exception("Bad timing type %s" % params.timing_type)
