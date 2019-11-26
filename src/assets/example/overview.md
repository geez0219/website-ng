# FastEstimator Application Hub

Welcome to FastEstimator Application Hub! Here we showcase different end-to-end AI examples implemented in FastEstimator. We will keep implementing new AI ideas and making state-of-the-art accessible to everyone.

## Purpose of Application Hub

* Provide place to learn implementation details of state-of-the-art
* Showcase FastEstimator functionalities in an end-to-end fashion
* Offer ready-made AI solutions for people to use in their own project/product


## Why not just learn from official implementation
If you ever spent time reading AI research papers, you will often find yourself asking: did I just spent 3 hours reading a paper where the underlying idea can be expressed in 3 minutes?

Similarly, people may use 5000+ lines of code or 500+ lines of code to implement the same idea using different AI frameworks. In FastEstimator, we strive to make things simpler and more intuitive while preserving the flexibility. As a result, many state-of-the-art AI implementations can be simplified greatly such that the code directly reflects the ideas. As an example, the [official implementation](https://github.com/tkarras/progressive_growing_of_gans) of [PGGAN](https://arxiv.org/abs/1710.10196) include 5000+ lines of code whereas [our implementation](https://github.com/fastestimator/fastestimator/tree/master/apphub/image_generation/pggan_nihchestxray) only uses 500+ lines.

To summarize, we spent time learning from the official implementation, so you can save time by learning from us!

## What's included in each example

Each example contains two files:

1. python file (.py): The FastEstimator source code needed to run the example.
2. jupyter notebook (.ipynb): notebook that provides step-by-step instructions and explanations about the implementation.


## How do I run each example

One can simply execute the python file of any example:
```
$ python lenet_mnist.py
```

or use our Command-Line Interface(CLI):

```
$ fastestimator train lenet_mnist.py
```

The benefit of CLI is allowing users to configure input args of `get_estimator`:

```
$ fastestimator train lenet_mnist.py --batch_size 64 --epochs 4
```

## Contributions
If you have implementations that we haven't done yet and want to join our efforts of making state-of-the-art AI easier, please consider contribute to us. We would really appreciate it! :smiley:
