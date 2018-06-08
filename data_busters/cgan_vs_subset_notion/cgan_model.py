"""
This module defines CGAN (Conditional Generative Adversarial Networks)
model.

Author: Nikolay Lysenko
"""


from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from data_busters.cgan_vs_subset_notion import batching 


def create_placeholders(
        n_items: int, z_dim: int
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Create placeholders for inputs and hyperparameters.

    :param n_items:
        number of unique items such that sets are constructed from them
    :param z_dim:
        number of dimensions (features) of random noise that is passed
        to generator
    :return:
        tuple of:
        * placeholder for discriminator input;
        * placeholder for generator input;
        * placeholder for learning rate
    """
    discriminator_input = tf.placeholder(tf.float32, (None, 2 * n_items))
    generator_input = tf.placeholder(tf.float32, (None, n_items + z_dim))
    learning_rate = tf.placeholder(tf.float32)
    return discriminator_input, generator_input, learning_rate


def discriminator(
        discriminator_input: tf.Tensor, layer_sizes: List[int],
        reuse: bool = False
        ) -> tf.Tensor:
    """
    Run input through discriminator network.

    :param discriminator_input:
        batch of objects that are real or fake results of concatenating
        a set from condition with a set pretending to be its superset
    :param layer_sizes:
        sizes of layers of the network
    :param reuse:
        if `True`, weights are reused
    :return:
        logits of the discriminator
    """
    transformed_input = discriminator_input
    with tf.variable_scope('discriminator', reuse=reuse):
        for layer_size in layer_sizes:
            transformed_input = tf.layers.dense(
                transformed_input,
                layer_size,
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.he_uniform()
            )
            transformed_input = tf.layers.batch_normalization(
                transformed_input,
                training=True  # Here, updates of moving stats never stop.
            )
            transformed_input = tf.nn.leaky_relu(transformed_input)
        logits = tf.layers.dense(
            transformed_input,
            1,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer()
        )
        # Actually, there can be `output = tf.sigmoid(logits)`,
        # but such `output` is not used.
    return logits


def generator(
        generator_input: tf.Tensor, n_items: int, layer_sizes: List[int],
        is_applied: bool = False
        ) -> tf.Tensor:
    """
    Run input through generator network.

    :param generator_input:
        input batch where each object is a result of concatenating
        a set from condition with random noise
    :param n_items:
        number of unique items such that sets are constructed from them
    :param layer_sizes:
        sizes of layers of the network
    :param is_applied:
        pass `True` if generator is not being trained (i.e., if it is
        used for inference)
    :return:
        generated output, a batch of fake sets pretending to be a
        supersets of sets from corresponding conditions
    """
    transformed_input = generator_input
    training = not is_applied
    with tf.variable_scope('generator', reuse=is_applied):
        for layer_size in layer_sizes:
            transformed_input = tf.layers.dense(
                transformed_input,
                layer_size,
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.he_uniform()
            )
            transformed_input = tf.layers.batch_normalization(
                transformed_input,
                training=training
            )
            transformed_input = tf.nn.leaky_relu(transformed_input)
        logits = tf.layers.dense(
            transformed_input,
            n_items,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer()
        )
        output = tf.sigmoid(logits)
    return output


def losses(
        real_input: tf.Tensor, generator_input: tf.Tensor,
        d_layer_sizes: List[int], g_layer_sizes: List[int]
        ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute the loss for the discriminator and the generator.

    :param real_input:
        batch of real objects for discriminator
    :param generator_input:
        batch that is generator input
    :param d_layer_sizes:
        sizes of layers of the discriminator
    :param g_layer_sizes:
        sizes of layers of the generator
    :return:
        discriminator loss and generator loss
    """
    n_items = real_input.get_shape().as_list()[1] // 2

    # Compute outputs.
    g_output = generator(generator_input, n_items, g_layer_sizes)
    fake_input = tf.concat((real_input[:, :n_items], g_output), axis=1)
    d_logits_real = discriminator(real_input, d_layer_sizes)
    d_logits_fake = discriminator(fake_input, d_layer_sizes, reuse=True)

    # Compute discriminator loss.
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_real,
            labels=tf.ones_like(d_logits_real)
        )
    )
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_fake,
            labels=tf.zeros_like(d_logits_fake)
        )
    )
    d_loss = d_loss_real + d_loss_fake

    # Compute generator loss.
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_fake,
            labels=tf.ones_like(d_logits_fake)
        )
    )

    return d_loss, g_loss


def optimization_operations(
        d_loss: tf.Tensor, g_loss: tf.Tensor, learning_rate: tf.Tensor,
        beta_one: float
        ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Return optimization operations for discriminator and generator.

    :param d_loss:
        discriminator loss
    :param g_loss:
        generator loss
    :param learning_rate:
        placeholder of learning rate
    :param beta_one:
        exponential decay rate for the first moment in the ADAM
        optimizer
    :return:
        discriminator training operation and
        generator training operation
    """
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta_one)
    g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta_one)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_op = d_optimizer.minimize(d_loss, var_list=d_vars)
        g_train_op = g_optimizer.minimize(g_loss, var_list=g_vars)
    return d_train_op, g_train_op


def evaluate_generator_output(
        sess: tf.Session, real_batch: np.ndarray, z_dim: int,
        generator_input: tf.Tensor, layer_sizes: List[int]
        ) -> type(None):
    """
    Print sets created by the generator and evaluate their
    plausibility.

    :param sess:
        session with (partially) trained generator
    :param real_batch:
        real batch such that sets to be generated are conditioned
        on conditions from it
    :param z_dim:
        number of dimensions (features) of random noise that is passed
        to generator
    :param generator_input:
        placeholder for generator input
    :param layer_sizes:
        sizes of layers of the network
    :return:
        None
    """
    generator_batch = batching.turn_into_generator_batch(real_batch, z_dim)
    n_items = real_batch.shape[1] // 2
    samples = sess.run(
        generator(generator_input, n_items, layer_sizes, is_applied=True),
        feed_dict={generator_input: generator_batch}
    )
    samples = np.around(samples).astype(np.int32)

    score = 0
    for i in range(samples.shape[0]):
        condition_items = real_batch[i, :n_items].tolist()
        print(f"Condition: {', '.join(condition_items)}")
        sample_items = samples[i, :].tolist()
        print(f"Sample: {', '.join(sample_items)}")
        if len(set(condition_items) - set(sample_items)) == 0:
            score += 1
    score /= samples.shape[0]
    print(f"Plausibility score is {score}.")


def train(
        df: pd.DataFrame, n_items: int, z_dim: int,
        d_layer_sizes: List[int], g_layer_sizes: List[int],
        n_epochs: int, batch_size: int, learning_rate: float, beta_one: float
        ) -> type(None):
    """
    Train generator and discriminator.

    :param df:
        dataframe with column 'sets'
    :param n_items:
        number of unique items such that sets are constructed from them
    :param z_dim:
        number of dimensions (features) of random noise that is passed
        to generator
    :param d_layer_sizes:
        sizes of layers of the discriminator
    :param g_layer_sizes:
        sizes of layers of the generator
    :param n_epochs:
        number of epochs (full passes through the dataset)
    :param batch_size:
        number of objects per batch (a group of objects such that
        weights are updated once per the group)
    :param learning_rate:
        learning rate for gradient-based optimization
    :param beta_one:
        exponential decay rate for the first moment in the ADAM
        optimizer
    :return:
        None
    """
    real_input, generator_input, lr = create_placeholders(n_items, z_dim)
    d_loss, g_loss = losses(
        real_input, generator_input, d_layer_sizes, g_layer_sizes
    )
    d_op, g_op = optimization_operations(d_loss, g_loss, lr, beta_one)

    batch_i = 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(n_epochs):
            real_batches = batching.yield_real_batches(
                df, batch_size, n_items
            )
            for real_batch in real_batches:
                # Training
                real_batch = batching.blur_real_batch(real_batch)
                generator_batch = batching.turn_into_generator_batch(
                    real_batch, z_dim
                )
                _ = sess.run(
                    d_op,
                    feed_dict={
                        real_input: real_batch,
                        generator_input: generator_batch,
                        lr: learning_rate
                    }
                )
                _ = sess.run(
                    g_op,
                    feed_dict={
                        real_input: real_batch,
                        generator_input: generator_batch,
                        lr: learning_rate
                    }
                )

                # Monitoring
                if batch_i % 100 == 0:
                    d_train_loss = d_loss.eval(
                        {
                            real_input: real_batch,
                            generator_input: generator_batch
                        }
                    )
                    g_train_loss = g_loss.eval(
                        {
                            real_input: real_batch,
                            generator_input: generator_batch
                        }
                    )
                    print(
                        f"Epoch: {epoch_i}: "
                        f"discriminator train loss: {d_train_loss}, "
                        f"generator train loss: {g_train_loss}"
                    )
                if batch_i % 1000 == 1:
                    evaluate_generator_output(
                        sess, real_batch[:10, :],
                        z_dim, generator_input, g_layer_sizes
                    )
                batch_i += 1
