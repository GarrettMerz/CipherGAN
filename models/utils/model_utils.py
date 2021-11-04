import tensorflow as tf


def collect_vars(fn):
  """Collect all new variables created within `fn`.

  Args:
    fn: a function that takes no arguments and creates trainable tf.compat.v1.Variable
      objects.

  Returns:
    outputs: the outputs of `fn()`.
    new_vars: a list of the newly created variables.
  """
  previous_vars = set(tf.compat.v1.trainable_variables())
  outputs = fn()
  current_vars = set(tf.compat.v1.trainable_variables())
  new_vars = current_vars.difference(previous_vars)
  return outputs, list(new_vars)


def get_embedding_var(params, name="input_emb", reuse=False):
  if reuse:
    return tf.compat.v1.contrib.framework.get_unique_variable("cycle_gan/" + name)
  else:
    return tf.compat.v1.Variable(
        tf.compat.v1.random_normal(
            [params.vocab_size, params.hidden_size], mean=0.0, stddev=1.0),
        name=name)


def embed_inputs(inputs, params, name="input_emb", reuse=False):
  assert inputs.dtype == tf.compat.v1.int64 or inputs.dtype == tf.compat.v1.int32, "Embedding lookup indices must be of integer type."
  w = get_embedding_var(params, name, reuse)
  return tf.compat.v1.gather(w, inputs)


def softmax_to_embedding(x, params):
  o_shape = x.shape.as_list()
  if o_shape[0] is None:
    o_shape[0] = tf.compat.v1.shape(x)[0]
  if o_shape[1] is None:
    o_shape[1] = tf.compat.v1.shape(x)[1]

  output_dist = tf.compat.v1.reshape(x, [o_shape[0] * o_shape[1], params.vocab_size])
  w_emb = get_embedding_var(params, reuse=True)
  output = tf.compat.v1.matmul(output_dist, w_emb)
  output = tf.compat.v1.reshape(output, [o_shape[0], o_shape[1], params.hidden_size])
  return output


def construct_vocab_lookup_table(vocab):
  mapping_string = tf.compat.v1.constant(vocab)
  return tf.compat.v1.contrib.lookup.index_to_string_table_from_tensor(
      mapping_string, default_value="<UNK>")


def log_text(F, G, params):
  lookup_table = construct_vocab_lookup_table(params.vocab)

  X_vocab = tf.compat.v1.expand_dims(tf.compat.v1.range(params.vocab_size), axis=0)
  if params.use_embeddings:
    X = embed_inputs(X_vocab, params, reuse=True)
  else:
    X = tf.compat.v1.one_hot(X_vocab, depth=params.vocab_size)
  X_map_distribution = F(X, params.F, params)
  X_map_indices = tf.compat.v1.argmax(X_map_distribution, axis=-1)
  # X_vocab = tf.compat.v1.Print(X_vocab, [X_vocab], message="X_vocab", summarize=10)
  # X_map_indices = tf.compat.v1.Print(
  #     X_map_indices, [X_map_indices], message="X_map_indices", summarize=10)
  X_map_text = lookup_table.lookup(tf.compat.v1.to_int64(X_map_indices))

  X_vocab_text = lookup_table.lookup(tf.compat.v1.to_int64(X_vocab))
  X_text = tf.compat.v1.string_join([X_vocab_text, "->", X_map_text])
  tf.compat.v1.summary.text("F_map", X_text)

  Y_vocab = tf.compat.v1.expand_dims(tf.compat.v1.range(params.vocab_size), axis=0)
  if params.use_embeddings:
    Y = embed_inputs(Y_vocab, params, reuse=True)
  else:
    Y = tf.compat.v1.one_hot(Y_vocab, depth=params.vocab_size)
  Y_map_distribution = G(Y, params.G, params)
  Y_map_indices = tf.compat.v1.argmax(Y_map_distribution, axis=-1)
  # Y_vocab = tf.compat.v1.Print(Y_vocab, [Y_vocab], message="Y_vocab", summarize=10)
  # Y_map_indices = tf.compat.v1.Print(
  #     Y_map_indices, [Y_map_indices], message="Y_map_indices", summarize=10)
  Y_map_text = lookup_table.lookup(tf.compat.v1.to_int64(Y_map_indices))

  Y_vocab_text = lookup_table.lookup(tf.compat.v1.to_int64(Y_vocab))
  Y_text = tf.compat.v1.string_join([Y_vocab_text, "->", Y_map_text])
  tf.compat.v1.summary.text("G_map", Y_text)


def groundtruth_accuracy(A, A_groundtruth, mask):
  groundtruth_mask = tf.compat.v1.to_float(mask)
  groundtruth_equalities = tf.compat.v1.to_float(tf.compat.v1.equal(A, A_groundtruth))
  groundtruth_accs = tf.compat.v1.reduce_sum(
      groundtruth_equalities * groundtruth_mask, axis=1) / tf.compat.v1.reduce_sum(
          groundtruth_mask, axis=1)
  return tf.compat.v1.reduce_mean(groundtruth_accs)


def sample_along_line(A_true, A_fake, params):
  A_unif = tf.compat.v1.tile(
        tf.compat.v1.random_uniform([params.batch_size, 1, 1]),
        [1, tf.compat.v1.shape(A_fake)[1], tf.compat.v1.shape(A_fake)[2]])

  return A_unif * A_fake + (1 - A_unif) * A_true


def wasserstein_penalty(discriminator, A_true, A_fake, params,
                        discriminator_params):
  A_interp = sample_along_line(A_true, A_fake, params)
  if params.use_embeddings:
    A_interp = softmax_to_embedding(A_interp, params)
  discrim_A_interp = discriminator(A_interp, discriminator_params, params)
  discrim_A_grads = tf.compat.v1.gradients(discrim_A_interp, [A_interp])
  discrim_A_grads = tf.compat.v1.squeeze(discrim_A_grads)

  if params.original_l2:
    l2_loss = tf.compat.v1.sqrt(
        tf.compat.v1.reduce_sum(
            tf.compat.v1.convert_to_tensor(discrim_A_grads)**2, axis=[1, 2]))
    if params.true_lipschitz:
      loss = params.wasserstein_loss * tf.compat.v1.reduce_mean(
          tf.compat.v1.nn.relu(l2_loss - 1)**2)
    else:
      loss = params.wasserstein_loss * tf.compat.v1.reduce_mean((l2_loss - 1)**2)
  else:
    loss = params.wasserstein_loss * (tf.compat.v1.nn.l2_loss(discrim_A_grads) - 1)**2
  return loss
