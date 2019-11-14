# Tutorial 3: Operator
___
In FastEstimator, the most important concept is the `Operator`, which is used extensively in `RecordWriter`, `Pipeline` and `Network`. In this tutorial, we're going to talk about everything you need to know about `Operator`.

Let's start with a short explanation: `Operator` is a class that works like a function, it is used in FastEstimator for constructing workflow graphs.

As we all know, a python function has 3 components: input variable(s), transformation logics and output variable(s). Similarly, an `Operator` has 3 parts: input key(s), a transformation function and output key(s). 

Now you may think: "`Operator` and function are almost the same, what's different between them? why do we need it?"

The difference is: a function uses variables whereas `Operator` uses keys (which is a representation of variable). The purpose of `Operator` is to allow users to construct a graph when variables are not created yet. In FastEstimator, we take care of the variable creation, routing and management, so that users can have a good night of sleep!

> This tutorial will first explain the concept of Operator, and then present an example of how to create and use it.

## How does Operator work?

Assuming our data is in a dictionary format with key-value pairs, and we have an `Operator` named `Add_one`, which adds 1 to the input. If our data is:
```python
data = {"x":1, "y":2}
```
and if we want to add 1 to the value associated with key `x`, we can simply do:

```python
Add_one(inputs="x", outputs="x")
```
At run time, the operator will:

1. take the value of the input key 'x' from the data dictionary
2. apply transformation functions to the value
3. write the output value to the data dictionary with output key 'x'

As a result, the data will become:
```python
{"x":2, "y":2}
```

Now let's add 1 to the value of `x` again and write the output to a new key `z`:
```python

Add_one(inputs="x", outputs="z")
```
our data then becomes:
```python
{"x":2, "y":2, "z":3}
```

## How to express Operator connections in FastEstimator?

`Operator` can take multiple inputs and produce multiple outputs. One can see the true power of `Operator` when combining multiple ones in a sequence. The Figure below lists several examples of graph topologies enabled by lists of `Operator`. We will talk about `Schedule` in detail in future tutorials.

<img src="image/ops.png">

## What different types of Operators are there?

On the implementation level, there are two types of `Operators` that every operator class inherits: `NumpyOp` and `TensorOp`. 

`NumpyOp` is used in the `ops` argument of `RecordWriter` only. Users can use any library inside the transformation function to calculate output. For example, users can call numpy, cv2, scipy functions etc. 

`TensorOp` is used in the `ops` argument of `Pipeline` and `Network`. Users are restricted to use tensor graph to construct the output. For example, the transformation logic has to be written in tensorflow graph.

## How is an Operator defined?

An `Operator` is defined like this:

```python
class Operator:
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def forward(self, data, state):
        return data
```

where:
 * `inputs` and `outputs` are the keys for input and outputs of `forward` function. 
 * `mode` is the execution mode ("train", "eval") that the Operator is active for. mode can be a string (like "train") or a list of string (like ["train", "eval"]). If mode is None, then it means Operator will be active for all scenarios.
 
If there are multiple inputs/outputs in `forward` function, `inputs` and `outputs` can be a list or tuple, the `forward` function's input/output variable `data` will have the same data type. For example, if we need `AddOne` to take multiple inputs:

```python

#assume dictionary is {"x":1, "y":2}

AddOne(inputs="x") #the data in forward function is 1

AddOne(inputs=["x", "y"]) #the data in forward function is [1,2]

AddOne(inputs=("x", "y")) #the data in forward function is (1,2)
```


## Operator demo in FastEstimator

We will now illustrate the usage of `Operator` in an end-to-end deep learning task.   
Let's start with the task shown in tutorial 2 and build more complex logics using `Operator`.

First, we have to generate some in-disk data images and csv files:


```python
# Import libraries 
import fastestimator as fe
import tensorflow as tf
import numpy as np
import os
```


```python
from fastestimator.dataset.mnist import load_data

# Download data in a temporary repository using load_data
train_csv, eval_csv, data_path = load_data()

print("image data is generated in {}".format(data_path))
```

### Step 0: Use pre-built Op and custom Op for data preprocessing in RecordWriter

In this example, when the csv files and training images are provided in-disk, we want to do two preprocessing steps upfront:

1. read the image (in grey scale) and its label. We will here use the pre-built `ImageReader` Operator in FastEstimator.
2. rescale the image. We want to reduce the pixel value range from [0, 255] to [-1, 1]. We will create a customized Operator to achieve this.


```python
from fastestimator.op import NumpyOp
from fastestimator.util import RecordWriter
from fastestimator.op.numpyop import ImageReader

# Create a custom Numpy Op to rescale images in forward function
class Rescale(NumpyOp):
    def forward(self, data, state):
        data = (data - 127.5) / 127.5
        return data

# Define the RecordWriter with two ops, Rescale and pre-defined ImageReader
writer = RecordWriter(save_dir=os.path.join(data_path, "tfrecords"),
                         train_data=train_csv,
                         validation_data=eval_csv,
                         ops=[ImageReader(inputs="x", parent_path=data_path, grey_scale=True), 
                              Rescale(outputs="x")])
```

Note that in the ops above, `ImageReader` does not have outputs and `Rescale` does not have inputs. This is because in FastEstimator, if the input of next operator uses the output of previous operator, then there is no need to read/write the data from/to the dictionary.

### Step 1: Use pre-built and custom Ops for Pipeline

As mentioned before, `Pipeline` is responsible for real-time preprocessing during the training (such as augmentation).
Let's do the following preprocessing during training for each batch:
1. Resize the image to (30,30), we are going to customize this operation.
2. Augment data with image rotation (-15 to +15 degree), we are going to use a pre-built operator for it. 

Some may argue that `Resize` can be done upfront in `RecordWriter`, which is indeed true.  But sometimes, resizing during training may have some benefits. For example, we can save disk space by storing 28x28 data instead of 30x30.  It is up to the user to choose based on his specific usecase.


```python
from fastestimator.op.tensorop import Augmentation2D
from fastestimator.op import TensorOp

# Create a custom Resize Tensor op
# We need init here as we want to add the size argument.
class Resize(TensorOp):
    def __init__(self, inputs, outputs, size):
        super().__init__(inputs=inputs, outputs=outputs)
        self.size = size
    
    def forward(self, data, state):
        data = tf.image.resize(data, self.size)
        return data

# Create Pipeline with Resize op and Augmentation pre-built op
# Augmentation2D automatically augment the dataset with rotation in the specified range.
pipeline = fe.Pipeline(data=writer,
                       batch_size=32,
                       ops=[Resize(inputs="x", size=(30, 30), outputs="x"),
                            Augmentation2D(outputs="x", mode="train", rotation_range=15)])
```

### Step 2: Use pre-built and custom ops for Network

Network is responsible for differentiable executions. Let's do the following:
1. feed the augmentated image to the network and get the predicted score
2. scale the predicted score 10 times and write it to a new key (this is only for demo purpose, it has no actual usage)



```python
from fastestimator.architecture import LeNet
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.op.tensorop.loss import SparseCategoricalCrossentropy

# Create a custom TensorOp
class Scale(TensorOp):
    def forward(self, data, state):
        data = data * 10
        return data

# Build the model and network
model = fe.build(model_def=lambda: LeNet(input_shape=(30, 30, 1)), model_name="lenet", optimizer="adam", loss_name="loss")
network = fe.Network(ops=[ModelOp(inputs="x", model=model, outputs="y_pred"), 
                          SparseCategoricalCrossentropy(y_pred="y_pred", y_true="y", outputs="loss"),
                          Scale(inputs="y_pred", outputs="y_pred_scaled")])
```

### Step 3: Create the Estimator and train!
Nothing different from tutorial here.


```python
# Create the estimator
estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=2)
```


```python
# Launch the training
estimator.fit()
```
