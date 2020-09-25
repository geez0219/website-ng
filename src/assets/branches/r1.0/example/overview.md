# FastEstimator Application Hub

Welcome to the FastEstimator Application Hub! Here we showcase different end-to-end AI examples implemented in FastEstimator. We will keep implementing new AI ideas and making state-of-the-art solutions accessible to everyone.

## Purpose of Application Hub

* Provide a place to learn implementation details of state-of-the-art solutions
* Showcase FastEstimator functionalities in an end-to-end fashion
* Offer ready-made AI solutions for people to use in their own projects/products

## Why not just learn from official implementations?

If you have ever spent time reading AI research papers, you will often find yourself asking: did I just spent 3 hours reading a paper where the underlying idea can be expressed in 3 minutes?

Similarly, people may use 5000 lines of code to implement an idea which could have been expressed in 500 lines using a different AI framework. In FastEstimator, we strive to make things simpler and more intuitive while preserving flexibility. As a result, many state-of-the-art AI implementations can be simplified greatly such that the code directly reflects the key ideas. As an example, the [official implementation](https://github.com/tkarras/progressive_growing_of_gans) of [PGGAN](https://arxiv.org/abs/1710.10196) includes 5000+ lines of code whereas [our implementation](https://github.com/fastestimator/fastestimator/blob/r1.0/apphub/image_generation/pggan/pggan_tf.py) requires less than 500.

To summarize, we spent time learning from the official implementation, so you can save time by learning from us!

## What's included in each example?

Each example contains three files:

1. A TensorFlow python file (.py): The FastEstimator source code needed to run the example with TensorFlow.
2. A PyTorch python file (.py): The FastEstimator source code needed to run the example with PyTorch.
3. A jupyter notebook (.ipynb): A notebook that provides step-by-step instructions and explanations about the implementation.

## How do I run each example

One can simply execute the python file of any example:
``` bash
$ python mnist_tf.py
```

Or use our Command-Line Interface (CLI):

``` bash
$ fastestimator train mnist_torch.py
```

One benefit of the CLI is that it allows users to configure the input args of `get_estimator`:

``` bash
$ fastestimator train lenet_mnist.py --batch_size 64 --epochs 4
```
