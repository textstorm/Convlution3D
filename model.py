
import tensorflow as tf
import numpy as np

class C3D(object):
  def __init__(self, args, sess, idx=0, name=None):
    self.img_h = args.img_h
    self.img_w = args.img_w
    self.frame_size = args.frame_size
    self.channels = args.channels
    self.nb_classes = args.nb_classes
    self.num_gpu = args.num_gpu
    self.sess = sess
    self.batch_size = args.batch_size

    self.images = tf.placeholder(tf.float32, [self.batch_size, self.frame_size, self.img_h, self.img_w, self.channels],
                                name="input_videos")
    self.labels = tf.placeholder(tf.int64, [self.batch_size], name='labels')
    self.is_train = tf.placeholder(tf.bool, name="is_train")
    self.dropout = tf.cond(self.is_train, lambda: args.dropout, lambda: 0.0)
    self.global_step = tf.get_variable(shape=[], initializer=tf.constant_initializer(0), 
                                       trainable=False, name='global_step')

    with tf.variable_scope(name):
      x = tf.nn.relu(self.conv3d(self.images, 3, 3, 3, 3, 64, "conv1"))
      # x = tf.nn.relu(tf.layers.conv3d(inputs=self.images, filters=64, kernel_size=3, padding='SAME'))
      x = self.max_pool3d(x, 1, "pool1")
      x = tf.nn.relu(self.conv3d(x, 3, 3, 3, 64, 128, "conv2"))
      x = self.max_pool3d(x, 2, "pool2")
      x = tf.nn.relu(self.conv3d(x, 3, 3, 3, 128, 256, "conv3a"))
      x = tf.nn.relu(self.conv3d(x, 3, 3, 3, 256, 256, "conv3b"))
      x = self.max_pool3d(x, 2, "pool3")
      x = tf.nn.relu(self.conv3d(x, 3, 3, 3, 256, 512, "conv4a"))
      x = tf.nn.relu(self.conv3d(x, 3, 3, 3, 512, 512, "conv4b"))
      x = self.max_pool3d(x, 2, "pool4")
      x = tf.nn.relu(self.conv3d(x, 3, 3, 3, 512, 512, "conv5a"))
      x = tf.nn.relu(self.conv3d(x, 3, 3, 3, 512, 512, "conv5b"))
      x = self.max_pool3d(x, 2, "pool5")
      # x = tf.transpose(x, perm=[0,1,4,2,3])
      x = tf.reshape(x, [self.batch_size, -1])
      x = tf.nn.relu(self.linear(x, 4096, name="fc1"))
      x = tf.nn.dropout(x, 1.0-self.dropout)
      x = tf.nn.relu(self.linear(x, 4096, name="fc2"))
      x = tf.nn.dropout(x, 1.0-self.dropout)
      logits = self.linear(x, self.nb_classes, name="logits")

    with tf.name_scope("loss"):
      cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=self.labels, logits=logits))
      tf.summary.scalar("loss" + '_cross_entropy', cross_entropy)
      self.total_loss = cross_entropy
      # weight_decay_loss = tf.reduce_mean(tf.get_collection('weightdecay_losses', scope="model_%d"%idx))
      # tf.summary.scalar("loss" + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss))
      # self.total_loss = cross_entropy + weight_decay_loss
      # tf.summary.scalar("loss" + '_total_loss', tf.reduce_mean(self.total_loss))

    with tf.name_scope("inference"):
      self.infer_op = tf.argmax(logits, 1)

    with tf.name_scope("accuracy"):
      correct_pred = tf.equal(tf.argmax(logits, 1), self.labels)
      self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    self.tvars = tf.trainable_variables()
    self.summary = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())

  def conv3d(self, x, fs, hs, ws, n_in, n_out, name=None):
    # w = self.variable_with_weight_decay(shape=[fs, hs, ws, n_in, n_out], name=name+"_w", wd=0.0005)
    w = self.variable_with_weight_decay(shape=[fs, hs, ws, n_in, n_out], name=name+"_w")
    b = self.variable_with_weight_decay(shape=[n_out], name=name+"_bias")
    x = tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', name=name)
    return tf.nn.bias_add(x, b)

  def linear(self, x, units, name=None):
    # w = self.variable_with_weight_decay(shape=[x.get_shape().as_list()[-1], units], name=name+"_w", wd=0.0005)
    w = self.variable_with_weight_decay(shape=[x.get_shape().as_list()[-1], units], name=name+"_w")
    b = self.variable_with_weight_decay(shape=[units], name=name+"_bias")
    return tf.nn.bias_add(tf.matmul(x, w), b)

  def max_pool3d(self, x, k, name):
    return tf.nn.max_pool3d(x, [1, k, 2, 2, 1], [1, k, 2, 2, 1], padding="SAME", name=name)

  def build_feed_dict(self, images, labels, is_train):
    return {self.images: images, self.labels: labels, self.is_train: is_train}

  def variable_on_cpu(self, name, shape, initializer):
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
    return var

  def variable_with_weight_decay(self, name, shape, wd=None):
    var = self.variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
      weight_decay = tf.nn.l2_loss(var) * wd
      tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

def get_multi_gpu_models(args, sess):
  models = []
  with tf.variable_scope(tf.get_variable_scope()):
    for gpu_idx in range(args.num_gpu):
      with tf.name_scope("model_%d"%gpu_idx), tf.device("/gpu:%d"%gpu_idx):
        c3d = C3D(args, sess, gpu_idx, name="C3D")
        tf.get_variable_scope().reuse_variables()
        models.append(c3d)
  return models

def restore_models(args, sess, models):
  new_models = []
  for model in models:
    restore_dict = restore_func(args.load_path, model.tvars)
    model.saver = tf.train.Saver(restore_dict)
    model.saver.restore(sess, args.load_path)
    new_models.append(model)
  return new_models

def restore_func(load_path, tvars):
  reader = tf.train.NewCheckpointReader(load_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  var_name = []
  for key in var_to_shape_map:
    var_name.append(key)
  var_name_sorted = sorted(var_name)

  convert_name = []
  for name in var_name_sorted:
    if "var_name" in name:
      name = name.replace("var_name", "C3D")
    if "bc" in name:
      name = name.replace("bc", "conv")
      name += "_bias"
    elif "bd" in name:
      name = name.replace("bd", "fc")
      name += "_bias"
    elif "bout" in name:
      name = name.replace("bout", "logits_bias")
      pass
    elif "wc" in name:
      name = name.replace("wc", "conv")
      name += "_w"
    elif "wd" in name:
      name = name.replace("wd", "fc")
      name += "_w"
    elif "wout" in name:
      name = name.replace("wout", "logits_w")
    else:
      pass
    convert_name.append(name)

  convert_dict = dict(zip(convert_name, var_name_sorted))
  restore_dict = dict()
  for v in tvars:
    tensor_name = v.name.split(':')[0]
    if reader.has_tensor(convert_dict[tensor_name]):
      # print('has tensor ', tensor_name)
      restore_dict[convert_dict[tensor_name]] = v
  restore_dict.pop("var_name/wout")
  restore_dict.pop("var_name/bout")
  # restore_dict.pop('var_name/bd1')
  # restore_dict.pop('var_name/bd2')
  # restore_dict.pop('var_name/wd1')
  # restore_dict.pop('var_name/wd2')
  return restore_dict

class MultiGPU(object):
  def __init__(self, args, models):
    self.model = models[0]
    self.max_grad_norm = args.max_grad_norm
    self.learning_rate = args.learning_rate
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.global_step = self.model.global_step
    self.summary = self.model.summary
    self.models = models

    loss_list = []
    grads_list = []
    accuracy = 0.0

    with tf.variable_scope(tf.get_variable_scope()):
      for gpu_idx, model in enumerate(self.models):
        with tf.name_scope("grads_%d"%gpu_idx), tf.device("/gpu:%d"%gpu_idx):
          loss = model.total_loss
          grads_and_vars = self.optimizer.compute_gradients(loss)
          grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
          loss_list.append(loss)
          grads_list.append(grads_and_vars)
          accuracy += model.accuracy
          tf.get_variable_scope().reuse_variables()

    self.loss = tf.add_n(loss_list) / float(len(loss_list))
    self.grads_and_vars = self.average_gradients(grads_list)
    self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
    self.accuracy = accuracy / len(self.models)

  def train(self, sess, images, labels):
    half_idx = len(images) // 2
    feed_dict = {}
    for idx, model in enumerate(self.models):
      images_feed = images[idx* half_idx: (idx+1)*half_idx]
      labels_feed = labels[idx* half_idx: (idx+1)*half_idx]
      feed_dict.update(model.build_feed_dict(images_feed, labels_feed, True))
    return sess.run([self.train_op, self.loss, self.accuracy, self.summary], feed_dict=feed_dict)

  def test(self, sess, input_x, sequence_length, input_y, keep_prob):
    return sess.run([self.loss, 
                     self.logits, 
                     self.batch_size], 
                     feed_dict={self.input_x: input_x, self.input_y: input_y, 
                               self.keep_prob: keep_prob, self.sequence_length: sequence_length})

  def average_gradients(self, grads_list):
    average_grads = []
    for grad_and_vars in zip(*grads_list):
      grads = []
      for g, _ in grad_and_vars:
        expanded_g = tf.expand_dims(g, 0)
        grads.append(expanded_g)
      grad = tf.concat(grads, 0)
      grad = tf.reduce_mean(grad, 0)
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

