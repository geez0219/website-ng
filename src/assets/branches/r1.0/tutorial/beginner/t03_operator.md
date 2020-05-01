# Tutorial 3: Operator

## Overview
In this tutorial we will talk about the following:
* **operator concept**
* **operator structure**
* **operator expression**
* **deep learning examples using operators**

## Operator Concept

In [tutorial 1](link_need), we know that the preprocessing in `Pipeline` and the training in `Network` can be divided into several sub-tasks:

* **Pipeline**: `Expand_dim` -> `Minmax`
* **Network**: `ModelOp` -> `CrossEntropy` -> `UpdateOp`

Each sub-task is a modular unit that takes inputs, performs an operation then produces outputs. As a result, we call them `Operator`, it is building block of `Pipeline` and `Network` API.

## Operator Structure

An Operator has 3 main components: 
* **inputs**: the key(s) of input data
* **outputs**: the key(s) of output data
* **forward function**: the transformation 

Implementation-wise, `Operator` is implemented as python class. Ignore `mode` for now as we will talk about `mode` extensively in [tutorial 9](link needed).


```python
class Op:
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode
    
    def forward(self, data, state):
        return data
```

## Operator Expression

In this section, we will demonstrate how different tasks can be concisely expressed in operators. 

### Single Operator
If the task only requires taking one feature then transform and overwrite the feature (e.g, `Minmax`), it can be expressed as:
<img src="assets/branches/r1.0/tutorial/../resources/t03_op_single1.png" alt="drawing" width="500"/>

If the task involves taking multiple features then overwrite them respectively (e.g, rotation of both image & mask), it can be expressed as:
<img src="assets/branches/r1.0/tutorial/../resources/t03_op_single2.png" alt="drawing" width="500"/>

### Multiple Operators
If there are two operators executing in sequential manner (e.g, `Minmax` followed by `Transpose`), it can be expressed as:
<img src="assets/branches/r1.0/tutorial/../resources/t03_op_multi1.png" alt="drawing" width="500"/>

Operator can easily handle more complicated data flows:
<img src="assets/branches/r1.0/tutorial/../resources/t03_op_multi2.png" alt="drawing" width="500"/>

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_multi3.png" alt="drawing" width="500"/>


## Deep learning with operator
In this section, we will show you how deep learning tasks can be modularized into combination of operators. Please note that the operator expression we provide in this section is simplified as pseudo-code, we will provide link to the actual python code for your reference.

### Image Classification:                                                                
[source](https://github.com/fastestimator/fastestimator/tree/1.0dev/apphub/image_classification/mnist)

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_cls.png" alt="drawing" width="800"/>

### DC-GAN:                                                                                  
[source](https://github.com/fastestimator/fastestimator/tree/1.0dev/apphub/image_generation/dcgan)

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_dcgan.png" alt="drawing" width="900"/>

### Adversarial Hardening:                                                                                  
[source](https://github.com/fastestimator/fastestimator/tree/1.0dev/apphub/adversarial_training/fgsm)

<img src="assets/branches/r1.0/tutorial/../resources/t03_op_adversarial.png" alt="drawing" width="900"/>



```python

```
