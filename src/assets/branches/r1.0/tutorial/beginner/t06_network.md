# Tutorial 6: Network 

## Overview
In this tutorial we are going to cover:
* [`Network` Scope](#t06network)
* [`TensorOp` and its Children](#t06tensorop)
* [How to Customize a `TensorOp`](#t06customize)
    * [TensorFlow](#t06tf)
    * [PyTorch](#t06torch)
    * [fe.backend](#t06backend)
* [Related Apphub Examples](#t06apphub)

<a id='t06network'></a>

## Network Scope
`Network` is one of the three main FastestEstimator APIs that defines not only a neural network model but also all of the operations to be performed on it. This can include the deep-learning model itself, loss calculations, model updating rules, and any other functionality that you wish to execute within a GPU. 
 
Here we show two `Network` example graphs to enhance the concept:

<img src="assets/branches/r1.0/tutorial/../resources/t06_network_example.png" alt="drawing" width="1000"/> 



As the figure shows, models (orange) are only piece of a `Network`. It also includes other operations such as loss computation (blue) and update rules (green) that will be used during the training process. 

<a id='t06tensorop'></a>

## TensorOp and its Children

A `Network` is composed of basic units called `TensorOps`. All of the building blocks inside a `Network` should derive from the `TensorOp` base class. A `TensorOp` is a kind of `Op` and therefore follows the same rules described in [tutorial 3](./tutorials/beginner/t03_operator). 

<img src="assets/branches/r1.0/tutorial/../resources/t06_tensorop_class.PNG" alt="drawing" width="500"/>

There are some common `TensorOp` classes we would like to specially mention because of their prevalence:

### ModelOp
Any model instance created from `fe.build` (see [tutorial 5](./tutorials/beginner/t05_model)) needs to be packaged as a `ModelOp` such that it can interact with other components inside the `Network` API. The orange blocks in the first figure are `ModelOps`.

### UpdateOp
FastEstimator use `UpdateOp` to associate the model with its loss. Unlike other `Ops` that use `inputs` and `outputs` for expressing their connections, `UpdateOp` uses the arguments `loss`, and `model` instead. The green blocks in the first figure are `UpdateOps`.

### Others (loss, gradient, etc.)
There are many ready-to-use `TensorOps` that users can directly import from `fe.op.tensorop`. Some examples include loss and gradient computation ops. For all available Ops please check out the FastEstimator API.


<a id='t06customize'></a>

## Customize a TensorOp
FastEstimator provides flexibility that allows users to customize their own `TensorOp`s by wrapping TensorFlow or PyTorch library calls, or by leveraging `fe.backend` API functions. Users only need to inherit the `TensorOp` class and overwrite its `forward` function.

If you want to customize a `TensorOp` by directly leveraging API calls from TensorFlow or PyTorch, **please make sure that all of the `TensorOp`s in the `Network` are backend-consistent**. In other words, you cannot have `TensorOp`s built specifically for TensorFlow and PyTorch in the same `Network`. Note that the `ModelOp` backend is determined by which library the model function uses, and so must be consistent with any custom `TensorOp` that you write.

Here we are going to demonstrate how to build a `TenorOp` that takes high dimensional inputs and returns an average scalar value.

<a id='t06tf'></a>

### Example Using TensorFlow


```python
from fastestimator.op.tensorop import TensorOp
import tensorflow as tf

class ReduceMean(TensorOp):
    def forward(self, data, state):
        return tf.reduce_mean(data)
```

<a id='t06torch'></a>

### Example Using PyTorch


```python
from fastestimator.op.tensorop import TensorOp
import torch

class ReduceMean(TensorOp):
    def forward(self, data, state):
        return torch.mean(data)
```

<a id='t06backend'></a>

### Example Using `fe.backend`
You don't need to worry about backend consistency if you import a FastEstimator-provided `TensorOp`, or customize your `TenosorOp` using the `fe.backend` API. FastEstimator auto-magically handles everything for you. 


```python
from fastestimator.op.tensorop import TensorOp
from fastestimator.backend import reduce_mean

class ReduceMean(TensorOp):
    def forward(self, data, state):
        return reduce_mean(data)
```

<a id='t06apphub'></a>

## Apphub Examples
You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [Fast Style Transfer](./examples/style_transfer/fst)
* [DC-GAN](./examples/image_generation/dcgan)
