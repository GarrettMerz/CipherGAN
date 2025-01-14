import tensorflow as tf

_INIT = dict()


def register(name):

  def add_to_dict(fn):
    global _INIT
    _INIT[name] = fn
    return fn

  return add_to_dict


def get_init(params):
  return _INIT[params.init_scheme](params)


@register("random")
def constant(params):
  return tf.compat.v1.random_normal_initializer()


@register("constant")
def constant(params):
  return tf.compat.v1.constant_initializer(0.1, tf.compat.v1.int32)
