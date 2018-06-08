# CGAN vs Subset Notion

## Overview

In this experiment, CGANs (Conditional Generative Adversarial Networks) are trained to solve a special problem. Namely, the goal is to build CGAN generating new sets such that a given set from condition is their subset.

## Dataset

A dataset is generated according to the following logic.

Let us consider sets that contain integer numbers from 1 to `k`. Every such set can be represented as a list of length `k` containing only zeros and ones. For example, if `k = 3`, set `{2}` can be represented as `[0, 1, 0]` and if `k = 4`, it can be represented as `[0, 1, 0, 0]`.

Each object from the dataset is represented with `2k` binary features. Also, each object from the dataset is sampled from a probabilistic distribution that imposes the following property: if the `i`-th feature (where `1 <= i <= k`) is 1, then the `(i + k)`-th feature must be 1 too. Actually, such property means that the set represented by the first `k` elements of the object representation is a subset of the set represented by the last `k` elements of the object representation.

The first `k` features are used for conditioning in CGANs under consideration. Given such partial description of an object, generator must sample the last `k` features of the object. To solve this problem correctly, CGAN must grasp the notion of subset.
