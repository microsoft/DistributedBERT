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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.python import debug as tf_debug

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "validation_input_file", None,
    "Input validation TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "input_dir", None,
    "Input TF example dir.")

flags.DEFINE_string(
    "validation_input_dir", None,
    "Input validation TF example dir.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_train_eval", False, "Whether to run train with eval.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", None, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


flags.DEFINE_integer("hooking_frequence", 100, "Hooking frequence.")

flags.DEFINE_bool("reduce_log", False, "Reduce log.")

flags.DEFINE_integer("keep_checkpoint_max", 1, "Keep checkpoint max.")

flags.DEFINE_bool("xla", True, "Whether to train with XLA optimization.")

flags.DEFINE_bool("adjust_lr", True, "Whether to adjust learning_rate.")

flags.DEFINE_integer("previous_train_steps", 0, "Previous train steps.")

flags.DEFINE_integer("post_train_steps", 0, "Post train steps.")

flags.DEFINE_bool("use_hvd", True, "Whether to use Horovod.")

flags.DEFINE_bool("use_compression", True, "Whether to use compression in Horovod.")

flags.DEFINE_bool("use_fp16", True, "Whether to use fp16.")

flags.DEFINE_bool("cos_decay", False, "Whether to use cos decay.")

flags.DEFINE_bool("use_lamb", False, "Whether to use lamb.")

flags.DEFINE_bool("auto_recover", False, "Whether to use auto recover.")

flags.DEFINE_string("recover_dir", None, "The output directory where the model checkpoints will be recovered.")

flags.DEFINE_integer("ckpt_no", None, "Checkpoint number of model to be recovered.")

flags.DEFINE_integer("ckpt_no_input", None, "Checkpoint number of input to be recovered.")

flags.DEFINE_bool("clip", False, "Whether to use clip.")

flags.DEFINE_bool("profile", False, "Whether to use profile.")


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, adjust_lr, use_hvd,
                     use_compression, use_fp16, clip, cos_decay,
                     use_lamb, previous_train_steps, post_train_steps):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        compute_type=tf.float16 if use_fp16 else tf.float32)

    (masked_lm_loss,
    masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
        bert_config, model.get_sequence_output(), model.get_embedding_table(),
        masked_lm_positions, masked_lm_ids, masked_lm_weights, clip)

    (next_sentence_loss, next_sentence_example_loss,
    next_sentence_log_probs) = get_next_sentence_output(
        bert_config, model.get_pooled_output(), next_sentence_labels, clip)

    total_loss = masked_lm_loss + next_sentence_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op, update_learning_rate = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, adjust_lr, use_hvd,
          use_compression, use_fp16, clip, cos_decay, use_lamb, previous_train_steps, post_train_steps)

      logging_hook = tf.train.LoggingTensorHook({"loss": total_loss, "learning_rate": update_learning_rate}, every_n_iter=FLAGS.hooking_frequence)
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = metric_fn(
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      )
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights, clip):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    if clip:
      log_probs = tf.log(tf.clip_by_value(tf.nn.softmax(logits, axis=-1), 1e-6, 1.0 - 1e-6))
    else:
      log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels, clip):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    if clip:
      log_probs = tf.log(tf.clip_by_value(tf.nn.softmax(logits, axis=-1), 1e-6, 1.0 - 1e-6))
    else:
      log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4,
                     batch_size=None,
                     use_hvd=True):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    # batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))

      if use_hvd:
        d = d.shard(hvd.size(), hvd.rank()) #TODO only for Horovod, shard to mimic single_GPU = False
        print("Data shard: %s %s" % (hvd.size(), hvd.rank()))

      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      # d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.use_hvd:
    hvd.init()

    if FLAGS.reduce_log and (hvd.rank() != 0):
      tf.logging.set_verbosity(tf.logging.ERROR)

    FLAGS.output_dir = FLAGS.output_dir if hvd.rank() == 0 else os.path.join(FLAGS.output_dir, str(hvd.rank()))

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_train_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  if FLAGS.recover_dir is not None:
    if FLAGS.use_hvd:
      FLAGS.recover_dir = FLAGS.recover_dir if hvd.rank() == 0 else os.path.join(FLAGS.recover_dir, str(hvd.rank()))
    path_ckpt = os.path.join(FLAGS.output_dir, "checkpoint")
    path_ckpt_input = os.path.join(FLAGS.output_dir, "checkpoint_input")

    if FLAGS.ckpt_no is not None and not tf.gfile.Exists(path_ckpt):
      with tf.gfile.GFile(path_ckpt, "w") as writer:
        writer.write('model_checkpoint_path: "%s-%s"\n' % (os.path.join(FLAGS.recover_dir, "model.ckpt"), str(FLAGS.ckpt_no)))
        writer.write('all_model_checkpoint_paths: "%s-%s"\n' % (os.path.join(FLAGS.recover_dir, "model.ckpt"), str(FLAGS.ckpt_no)))

    if FLAGS.ckpt_no_input is not None and not tf.gfile.Exists(path_ckpt_input):
      with tf.gfile.GFile(path_ckpt_input, "w") as writer:
        writer.write('model_checkpoint_path: "%s-%s"\n' % (os.path.join(FLAGS.recover_dir, "input.ckpt"), str(FLAGS.ckpt_no_input)))
        writer.write('all_model_checkpoint_paths: "%s-%s"\n' % (os.path.join(FLAGS.recover_dir, "input.ckpt"), str(FLAGS.ckpt_no_input)))

  if FLAGS.use_hvd and hvd.rank() == 0 and (FLAGS.do_train or FLAGS.do_train_eval):
    (cpath, cname) = os.path.split(FLAGS.bert_config_file)
    tf.gfile.Copy(FLAGS.bert_config_file, os.path.join(FLAGS.output_dir, cname), True)

  input_files = []
  if FLAGS.input_file is not None:
    for input_pattern in FLAGS.input_file.split(","):
      input_files.extend(tf.gfile.Glob(input_pattern))
  if FLAGS.input_dir is not None:
    for filename in tf.gfile.ListDirectory(FLAGS.input_dir):
      input_files.extend(tf.gfile.Glob(os.path.join(FLAGS.input_dir, filename)))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  validation_input_files = []
  if FLAGS.validation_input_file is None and FLAGS.validation_input_dir is None:
    validation_input_files = input_files
  else:
    if FLAGS.validation_input_file is not None:
      for input_pattern in FLAGS.validation_input_file.split(","):
        validation_input_files.extend(tf.gfile.Glob(input_pattern))
    if FLAGS.validation_input_dir is not None:
      for filename in tf.gfile.ListDirectory(FLAGS.validation_input_dir):
        validation_input_files.extend(tf.gfile.Glob(os.path.join(FLAGS.validation_input_dir, filename)))

  tf.logging.info("*** Input Validation Files ***")
  for input_file in validation_input_files:
    tf.logging.info("  %s" % input_file)

  config = tf.ConfigProto()
  if FLAGS.xla:
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  if FLAGS.use_hvd:
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.allow_growth=True

  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      log_step_count_steps=FLAGS.hooking_frequence,
      session_config=config)

  if FLAGS.use_hvd and hvd.rank() != 0 and not FLAGS.auto_recover:
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        save_checkpoints_steps=None,
        save_checkpoints_secs=None,
        log_step_count_steps=FLAGS.hooking_frequence,
        session_config=config)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      adjust_lr=FLAGS.adjust_lr,
      use_hvd=FLAGS.use_hvd,
      use_compression=FLAGS.use_compression,
      use_fp16=FLAGS.use_fp16,
      clip=FLAGS.clip,
      cos_decay=FLAGS.cos_decay,
      use_lamb=FLAGS.use_lamb,
      previous_train_steps=FLAGS.previous_train_steps,
      post_train_steps=FLAGS.post_train_steps)

  hooks = []

  if FLAGS.use_hvd:
    hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    if hvd.rank() == -1: #if debug, set 0
      CLIDebugHook = tf_debug.LocalCLIDebugHook(ui_type='readline')
      CLIDebugHook.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
      hooks.append(CLIDebugHook)

    if FLAGS.profile and hvd.rank() == 0:
      ProfilerHook = tf.train.ProfilerHook(save_steps=FLAGS.hooking_frequence, output_dir=FLAGS.output_dir, show_dataflow=True, show_memory=True)
      hooks.append(ProfilerHook)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True,
        batch_size=FLAGS.train_batch_size,
        use_hvd=FLAGS.use_hvd)

    if FLAGS.auto_recover:
      hooks.append(tf.data.experimental.CheckpointInputPipelineHook(estimator))

    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps, hooks=hooks)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder(
        input_files=validation_input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False,
        batch_size=FLAGS.eval_batch_size,
        use_hvd=FLAGS.use_hvd)

    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_train_eval:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True,
        batch_size=FLAGS.train_batch_size,
        use_hvd=FLAGS.use_hvd)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    eval_input_fn = input_fn_builder(
        input_files=validation_input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False,
        batch_size=FLAGS.eval_batch_size,
        use_hvd=FLAGS.use_hvd)

    if FLAGS.auto_recover:
      hooks.append(tf.data.experimental.CheckpointInputPipelineHook(estimator))

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
  # flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
