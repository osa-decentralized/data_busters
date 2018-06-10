"""
This module generates dataset and provides some utilities for
training (such as batching and blurring).

Author: Nikolay Lysenko
"""


from typing import Generator

import numpy as np


def generate_data(size: int, n_items: int) -> np.ndarray:
    """
    Generate synthetic dataset described in `README.md` file from
    the current directory.

    :param size:
        number of objects to generate
    :param n_items:
        number of unique items such that sets are constructed from them
    :return:
         array of shape (`size`, 2 * `n_items`)
    """
    conditions = np.random.binomial(n=1, p=0.5, size=(size, n_items))
    extra_items = np.random.binomial(n=1, p=0.5, size=(size, n_items))
    supersets = np.maximum(conditions, extra_items)
    dataset = np.hstack((conditions, supersets))
    return dataset


def yield_real_batches(
        dataset: np.ndarray, batch_size: int
        ) -> Generator[np.ndarray, None, None]:
    """
    Yield batches with real examples for discriminator.
    The last batch is dropped if it is not of full size.

    :param dataset:
        synthetic dataset
    :param batch_size:
        number of objects per batch
    :yield:
         batches of shape (batch size, width of dataset)
    """
    size_of_incomplete_batch = dataset.shape[0] % batch_size
    dataset = dataset[:-size_of_incomplete_batch, :]
    dataset = dataset.reshape(-1, batch_size, dataset.shape[1])
    for i in range(dataset.shape[0]):
        yield dataset[i]


def turn_into_generator_batch(
        real_batch: np.ndarray, z_dim: int
        ) -> np.ndarray:
    """
    Make a batch for generator from a real batch for discriminator.

    :param real_batch:
        batch of shape ('batch_size`, 2 * `n_items`)
        where columns for conditioning go first and columns
        with supersets go last
    :param z_dim:
        dimensionality (number of columns) of noise
    :return:
        batch with the same condition, but noise instead of
        initial supersets
    """
    z_batch_shape = (real_batch.shape[0], z_dim)
    z_batch = np.random.normal(size=z_batch_shape)
    condition_size = real_batch.shape[1] // 2
    generator_batch = np.hstack((real_batch[:, :condition_size], z_batch))
    return generator_batch


def blur_real_batch(
        real_batch: np.ndarray, strength: float = 0.1
        ) -> np.ndarray:
    """
    Make values of `real_batch` float.
    This prevents discriminator from learning to distinguish
    integer values from float values.

    :param real_batch:
        batch of shape ('batch_size`, 2 * `n_items`)
        where columns for conditioning go first and columns
        with supersets go last
    :param strength:
        value between 0 and 0.5; the higher it is, the more
        blurred results are
    :return:
        batch where values are float
    """
    if strength < 0 or strength > 0.5:
        raise ValueError('Argument `strength` must be between 0 and 0.5.')
    condition_size = real_batch.shape[1] // 2
    conditions = real_batch[:, :condition_size]
    supersets = real_batch[:, condition_size:]
    noise = np.random.uniform(0, 0.1, supersets.shape)
    blurred_supersets = np.absolute(supersets - noise)
    blurred_real_batch = np.hstack((conditions, blurred_supersets))
    return blurred_real_batch
