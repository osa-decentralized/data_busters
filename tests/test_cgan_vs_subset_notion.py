"""
This script tests functionality related to the experiment named
'CGAN vs Subset Notion'.

Author: Nikolay Lysenko
"""


import unittest

import numpy as np

from data_busters.cgan_vs_subset_notion import training_set


class TestTrainingSet(unittest.TestCase):
    """
    Tests for functions from `training_set` module.
    """

    def setUp(self) -> type(None):
        """
        Reset random state before each test.

        :return:
            None
        """
        np.random.seed(361)

    def test_generate_data(self) -> type(None):
        """
        Test `generate_data` function.

        :return:
            None
        """
        size = 5
        n_items = 4
        arr = training_set.generate_data(size, n_items)
        self.assertEqual(arr.shape, (size, 2 * n_items))
        self.assertEqual(np.unique(arr).tolist(), [0, 1])
        self.assertEqual((arr[:, n_items:] - arr[:, :n_items]).min(), 0)

    def test_yield_real_batches(self) -> type(None):
        """
        Test `yield_real_batches` function.

        :return:
            None
        """
        arr = np.array([
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0]
        ])
        batch_list = list(training_set.yield_real_batches(arr, batch_size=2))
        true_batch_list = [
            np.array([
                [0, 1, 0, 1, 1, 0],
                [1, 0, 0, 1, 1, 0]
            ]),
            np.array([
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 1]
            ])
        ]
        self.assertEqual(len(batch_list), len(true_batch_list))
        for arr, true_arr in zip(batch_list, true_batch_list):
            np.testing.assert_equal(arr, true_arr)

    def test_turn_into_generator_batch(self) -> type(None):
        """
        Test `turn_into_generator_batch` function.

        :return:
            None
        """
        real_batch = np.array([
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        generator_batch = training_set.turn_into_generator_batch(
            real_batch, z_dim=2
        )
        self.assertEqual(generator_batch.shape, (3, 5))
        true_batch = np.array([
            [0, 1, 0, 0.33144363, -0.66543322],
            [1, 0, 0, -0.23577372, 0.98548883],
            [0, 0, 0, 1.38739009, 1.70285296]
        ])
        np.testing.assert_allclose(generator_batch, true_batch)

    def test_blur_real_batch(self) -> type(None):
        """
        Test `blur_real_batch` function.

        :return:
            None
        """
        real_batch = np.array([
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        blurred_batch = training_set.blur_real_batch(real_batch)
        one_hots = (blurred_batch > 0.5).astype(int)
        np.testing.assert_equal(real_batch, one_hots)
        true_blurred_batch = np.array([
            [0, 1, 0, 0.98898011, 0.93058451, 0.0876186],
            [1, 0, 0, 0.95900008, 0.93839625, 0.05945409],
            [0, 0, 0, 0.04102986, 0.9493844, 0.07379478]
        ])
        np.testing.assert_allclose(blurred_batch, true_blurred_batch)


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [TestTrainingSet()]
    for tester in testers:
        suite = test_loader.loadTestsFromModule(tester)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()
