# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
# pylint: enable=g-direct-tensorflow-import

import horovod.tensorflow as hvd


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu, adjust_lr, use_hvd,
                     use_compression, use_fp16, clip, cos_decay, use_lamb=False,
                     previous_train_steps=0, post_train_steps=0):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # if adjust_lr:
  #   # avoid step change in learning rate at end of warmup phase
  #   decayed_learning_rate_at_crossover_point = init_lr * (1.0-float(num_warmup_steps)/float(num_train_steps + previous_train_steps + post_train_steps))
  #   adjusted_init_lr = init_lr * (init_lr / decayed_learning_rate_at_crossover_point)
  #   print('decayed_learning_rate_at_crossover_point = %e, adjusted_init_lr = %e' % (decayed_learning_rate_at_crossover_point, adjusted_init_lr))

  #   learning_rate = tf.constant(value=adjusted_init_lr, shape=[], dtype=tf.float32)

  if cos_decay:
    # Implements cosine decay of the learning rate.
    learning_rate = tf.train.cosine_decay(
        learning_rate,
        global_step + previous_train_steps - (num_warmup_steps if adjust_lr else 0),
        num_train_steps + previous_train_steps + post_train_steps - (num_warmup_steps if adjust_lr else 0),
        alpha=0.0)
  else:
    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step + previous_train_steps - (num_warmup_steps if adjust_lr else 0),
        num_train_steps + previous_train_steps + post_train_steps - (num_warmup_steps if adjust_lr else 0),
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step + previous_train_steps, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  if use_lamb:
    optimizer = LAMBOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  else:
    optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  tf.summary.scalar("learning_rate", optimizer.learning_rate)

  if use_hvd:
    optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=True, compression=hvd.Compression.fp16 if use_compression else hvd.Compression.none)

  if use_fp16:
    loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)

  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  all_are_finite = tf.constant(True, dtype=tf.bool)
  tvars = tf.trainable_variables()
  if use_hvd:
    grads_and_vars = optimizer.compute_gradients(loss, tvars)
    grads_and_vars = [(g,v) for g,v in grads_and_vars if g is not None]
    grads, tvars = list(zip(*grads_and_vars))
    all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads]) if use_fp16 else tf.constant(True, dtype=tf.bool)
  else:
    grads = tf.gradients(loss, tvars)

  # This is how the model was pre-trained.
  # ensure global norm is a finite number
  # to prevent clip_by_global_norm from having a hizzy fit.
  (clipped_grads, _) = tf.clip_by_global_norm(
        grads, clip_norm=1.0,
        use_norm=tf.cond(
            all_are_finite,
            lambda: tf.global_norm(grads),
            lambda: tf.constant(1.0)))

  train_op = optimizer.apply_gradients(
      list(zip(clipped_grads, tvars)), global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.
  if clip:
    new_global_step = global_step + 1
  else:
    new_global_step = tf.cond(all_are_finite, lambda: global_step+1, lambda: global_step)
    new_global_step = tf.identity(new_global_step, name='step_update')
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op, learning_rate


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = tf.identity(learning_rate, name='learning_rate')
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


class LAMBOptimizer(tf.train.Optimizer):
  """LAMB (Layer-wise Adaptive Moments optimizer for Batch training)."""
  # A new optimizer that includes correct L2 weight decay, adaptive
  # element-wise updating, and layer-wise justification. The LAMB optimizer
  # was proposed by Yang You, Jing Li, Jonathan Hseu, Xiaodan Song,
  # James Demmel, and Cho-Jui Hsieh in a paper titled as Reducing BERT
  # Pre-Training Time from 3 Days to 76 Minutes (arxiv.org/abs/1904.00962)

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               name="LAMBOptimizer"):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    # TODO(jingli): validate if exclude_from_layer_adaptation is necessary.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = linalg_ops.norm(param, ord=2)
        g_norm = linalg_ops.norm(update, ord=2)
        ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
            math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      update_with_lr = ratio * self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
