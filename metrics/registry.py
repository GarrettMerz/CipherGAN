import tensorflow as tf

_METRICS = dict()


def register(name, keys=[None]):

  def add_to_dict(fn):
    global _METRICS
    _METRICS[name] = (fn, keys)
    return fn

  return add_to_dict


def get_metrics(metrics, hparams, key=None):
  return {
      metric_name + (key if key is not None else ""):
      _METRICS[metric_name][0](hparams, key)
      for metric_name in metrics.split("-") for key in _METRICS[metric_name][1]
  }


@register("ce")
def get_cross_entropy(hparams, key=None):

  def _cross_entropy(predictions, labels, weights=1.0):
    return tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=labels,
        logits=predictions,
        weights=weights,
        label_smoothing=hparams.label_smoothing)

  return tf.compat.v1.contrib.learn.MetricSpec(
      _cross_entropy, prediction_key=key, label_key=key)


@register("mse")
@register("xy_mse", ["X", "Y"])
def get_mean_squared_error(hparams, key=None):

  def _mean_squared_error(predictions, labels, weights=1.0):
    return tf.compat.v1.losses.mean_squared_error(
        labels=labels, predictions=predictions, weights=weights)

  return tf.compat.v1.contrib.learn.MetricSpec(
      _mean_squared_error, prediction_key=key, label_key=key)


@register("shift")
def get_shift_error(hparams, key=None):

  def _shift_error(predictions, labels, weights=1.0):
    predictions = tf.compat.v1.reshape(predictions, [hparams.batch_size, -1])
    labels = tf.compat.v1.reshape(labels, [hparams.batch_size, -1])

    elements_equal = tf.compat.v1.equal(
        tf.compat.v1.to_int32(tf.compat.v1.round(predictions)), tf.compat.v1.to_int32(labels))
    sequence_equal = tf.compat.v1.reduce_all(elements_equal, axis=1)
    return tf.compat.v1.reduce_mean(tf.compat.v1.to_float(sequence_equal))

  return tf.compat.v1.contrib.learn.MetricSpec(
      _shift_error, prediction_key=key, label_key=key)


@register("acc")
def get_accuracy(hparams, key=None):
  return tf.compat.v1.contrib.learn.MetricSpec(
      tf.compat.v1.contrib.metrics.accuracy, prediction_key=key, label_key=key)
