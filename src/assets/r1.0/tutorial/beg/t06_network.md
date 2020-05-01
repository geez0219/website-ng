# Tutorial 6: Network 

## Overview
In this tutorial we are going to cover:
* `Network` scope
* `TensorOp` and its inherited class
* How to Customize `TensorOp`
    * TensorFlow
    * Pytorch
    * fe.backend

## Network scope
`Network` is one of the three FastestEstimator main APIs that defines not only network model but all operation involved with it. The scope includes deep learning model, loss calculation, model updating units and all other logics that you wish to execute inside GPU. 
 
Here we shows two `Network` example graphs to enhance the Network concept.

<img src="assets/tutorial/../resources/t06_network_example.png" alt="drawing" width="1000"/> 



As the figure shown, models (orange) are only part of the `Network`. It also includes other operations connecting with them like loss computation (blue) and update units (geen) that will be used in training process. 

## TensorOp and its inherited class

`Network` is composed by its basic components --- `TensorOps` which means all blocks inside `Network` should either be `TensorOp` or class that inherit from `TensorOp`. `TensorOp` is a kind of `Op` and therefore follows its connecting rule. 

There are some common TensorOp-inherit classes we like to specially bring up because of their prevalence. 
<img src="assets/tutorial/../resources/t06_tensorop_class.PNG" alt="drawing" width="500"/>

### ModelOp
Any model instance created from `fe.build` (reference: **tutorial 5**) needs to be packaged as a `ModelOp` such that it can interact with other components inside `Network` API. The orange blocks in the first figure are `ModelOps`.

### UpdateOp
FastEstimator use `UpdateOp` to associate the model with its loss. Unlike other `Ops` that use `inputs`, `outputs` for expressing connection, UpdateOp uses argument `loss`, and `model` instead. The green blocks in the first figure are `UpdateOps`.

### Others (loss, gradient...)
There are many read-to-use TensorOps that users can directly import from `fe.op.tensorop` like operations about loss and gradient computation. For all available Ops please check out Fastestimator API.


## Customize TensorOp
FastEstimator provides flexibility that allows users to customize their own `TensorOp` from TensorFlow and Pytorch library or from `fe.backend` API function. Users only need inherit `TensorOps` class and overwrite its `forward` function.

If users want to customize `TensorOp` using TensorFlow or Pytorch library, **please make sure all `TensorOp` in the `Network` need to be backend-consistent**. This means they cannot have `TensorOps` built from TensorFlow and Pytorch in the same `Network`. `ModelOps`'s backend is determined by which library the model function uses. (Reminder: `ModelOp` is also an `TensorOp` so it also needs to be backend-consistent)

Here we are going to demonstrate building a `TenorOp` logic that takes high dimension input and output a average scalar value.

### Example for using Tensorflow
Tensorflow already has the function. 


```python
from fastestimator.op.tensorop import TensorOp
import tensorflow as tf

class ReduceMean(TensorOp):
    def forward(self, data, state):
        return tf.reduce_mean(data)
```

### Example for using Pytorch
Pytorch already has the function.


```python
from fastestimator.op.tensorop import TensorOp
import torch

class ReduceMean(TensorOp):
    def forward(self, data, state):
        return torch.mean(data)
```

### Example for using `fe.backend`
Users don't need to handle the backend issue if they use imported `TensorOp` or customize their `TenosorOp` using `fe.backend` API function because they are design to handle both backend.    



```python
from fastestimator.op.tensorop import TensorOp
from fastestimator.backend import reduce_mean

class ReduceMean(TensorOp):
    def forward(self, data, state):
        return reduce_mean(data)
```
