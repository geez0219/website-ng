# Tutorial 3: Operator

## Overview
In this tutorial we will introduce the `Operator` - a fundamental building block within FastEstimator. This tutorial is structured as follows:

* [Operator Definition](./tutorials/beginner/t03_operator#t03Def)
* [Operator Structure](./tutorials/beginner/t03_operator#t03Structure)
* [Operator Expression](./tutorials/beginner/t03_operator#t03Exp)
* [Deep Learning Examples using Operators](./tutorials/beginner/t03_operator#t03DL)

<a id='t03Def'></a>

## Operator Definition

From [tutorial 1](./tutorials/beginner/t01_getting_started), we know that the preprocessing in `Pipeline` and the training in `Network` can be divided into several sub-tasks:

* **Pipeline**: `Expand_dim` -> `Minmax`
* **Network**: `ModelOp` -> `CrossEntropy` -> `UpdateOp`

Each sub-task is a modular unit that takes inputs, performs an operation, and then produces outputs. We therefore call these sub-tasks `Operator`s, and they form the building blocks of the FastEstimator `Pipeline` and `Network` APIs.

<a id='t03Structure'></a>

## Operator Structure

An Operator has 3 main components: 
* **inputs**: the key(s) of input data
* **outputs**: the key(s) of output data
* **forward function**: the transformation to be applied

The base class constructor also takes a `mode` argument, but for now we will ignore it since `mode` will be discussed extensively in [tutorial 9](./tutorials/beginner/t09_inference).


```python
class Op:
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode
    
    def forward(self, data, state):
        return data
```

<a id='t03Exp'></a>

## Operator Expression

In this section, we will demonstrate how different tasks can be concisely expressed in operators. 

### Single Operator
If the task only requires taking one feature as input and transforming it to overwrite the old feature (e.g, `Minmax`), it can be expressed as:

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_single1.png" alt="drawing" width="500"/>

If the task involves taking multiple features and overwriting them respectively (e.g, rotation of both an image and its mask), it can be expressed as:

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_single2.png" alt="drawing" width="500"/>

### Multiple Operators
If there are two `Operator`s executing in a sequential manner (e.g, `Minmax` followed by `Transpose`), it can be expressed as:

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_multi1.png" alt="drawing" width="500"/>

`Operator`s can also easily handle more complicated data flows:

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_multi2.png" alt="drawing" width="500"/>

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_multi3.png" alt="drawing" width="500"/>


<a id='t03DL'></a>

## Deep Learning Examples using Operators

In this section, we will show you how deep learning tasks can be modularized into combinations of `Operator`s. Please note that the `Operator` expressions we provide in this section are essentially pseudo-code. Links to full python examples are also provided.

### Image Classification:                                                                
[MNIST](./examples/image_classification/mnist)

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_cls.png" alt="drawing" width="800"/>

### DC-GAN:                                                                                  
[DC-GAN](./examples/image_generation/dcgan)

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_dcgan.png" alt="drawing" width="900"/>

### Adversarial Hardening:                                                                                  
[FGSM](./examples/adversarial_training/fgsm)

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_adversarial.png" alt="drawing" width="900"/>

