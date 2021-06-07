# Tutorial 5: Model

## Overview

In this tutorial we will cover:

* [Instantiating and Compiling a Model](tutorials/r1.0/beginner/t05_model/#t05compile)
* [The Model Function](tutorials/r1.0/beginner/t05_model/#t05model)
    * [Custom Models](tutorials/r1.0/beginner/t05_model/#t05custom)
    * [FastEstimator Models](tutorials/r1.0/beginner/t05_model/#t05fe)
    * [Pre-Trained Models](tutorials/r1.0/beginner/t05_model/#t05trained)
* [The Optimizer Function](tutorials/r1.0/beginner/t05_model/#t05optimizer)
* [Loading Model Weights](tutorials/r1.0/beginner/t05_model/#t05weights)
* [Specifying a Model Name](tutorials/r1.0/beginner/t05_model/#t05name)
* [Related Apphub Examples](tutorials/r1.0/beginner/t05_model/#t05apphub)

<a id='t05compile'></a>

## Instantiating and Compiling a model

We need to specify two things to instantiate and compile a model:
* model_fn
* optimizer_fn

Model definitions can be implemented in Tensorflow or Pytorch and instantiated by calling **`fe.build`** which constructs a model instance and associates it with the specified optimizer.

<a id='t05model'></a>

## Model Function

`model_fn` should be a function/lambda function which returns either a `tf.keras.Model` or `torch.nn.Module`. FastEstimator provides several ways to specify the model architecture:

* Custom model architecture
* Importing a pre-built model architecture from FastEstimator
* Importing pre-trained models/architectures from PyTorch or TensorFlow

<a id='t05custom'></a>

### Custom model architecture
Let's create a custom model in TensorFlow and PyTorch for demonstration.

#### tf.keras.Model


```python
import fastestimator as fe
import tensorflow as tf
from tensorflow.keras import layers

def my_model_tf(input_shape=(30, ), num_classes=2):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    return model

model_tf = fe.build(model_fn=my_model_tf, optimizer_fn="adam")
```

#### torch.nn.Module


```python
import torch
import torch.nn as nn
import torch.nn.functional as fn

class my_model_torch(nn.Module):
    def __init__(self, num_inputs=30, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(num_inputs, 32), 
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(32, 8), 
                                    nn.ReLU(inplace=True),
                                    nn.Linear(8, num_classes))

    def forward(self, x):
        x = self.layers(x)
        x_label = torch.softmax(x, dim=-1)
        return x_label

    
model_torch = fe.build(model_fn=my_model_torch, optimizer_fn="adam")
```

<a id='t05fe'></a>

### Importing model architecture from FastEstimator

Below we import a PyTorch LeNet architecture from FastEstimator. See our [Architectures](assets/branches/r1.0/fastestimator/architecture) folder for a full list of the architectures provided by FastEstimator.


```python
from fastestimator.architecture.pytorch import LeNet
# from fastestimator.architecture.tensorflow import LeNet  # One can also use a TensorFlow model

model = fe.build(model_fn=LeNet, optimizer_fn="adam")
```

<a id='t05trained'></a>

### Importing pre-trained models/architectures from PyTorch or TensorFlow

Below we show how to define a model function using a pre-trained resnet model provided by TensorFlow and PyTorch respectively. We load the pre-trained models using a lambda function.

#### Pre-trained model from tf.keras.applications 


```python
resnet50_tf = fe.build(model_fn=lambda: tf.keras.applications.ResNet50(weights='imagenet'), optimizer_fn="adam")
```

#### Pre-trained model from torchvision 


```python
from torchvision import models

resnet50_torch = fe.build(model_fn=lambda: models.resnet50(pretrained=True), optimizer_fn="adam")
```

<a id='t05optimizer'></a>

## Optimizer function

`optimizer_fn` can be a string or lambda function.

### Optimizer from String
Specifying a string for the `optimizer_fn` loads the optimizer with default parameters. The optimizer strings accepted by FastEstimator are as follows:
- Adadelta: 'adadelta'
- Adagrad: 'adagrad'
- Adam: 'adam'
- Adamax: 'adamax'
- RMSprop: 'rmsprop'
- SGD: 'sgd'

### Optimizer from Function

To specify specific values for the optimizer learning rate or other parameters, we need to pass a lambda function to the `optimizer_fn`.


```python
# TensorFlow 
model_tf = fe.build(model_fn=my_model_tf, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))

# PyTorch
model_torch = fe.build(model_fn=my_model_torch, optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=1e-4))
```

If a model function returns multiple models, a list of optimizers can be provided. See the **[pggan apphub](examples/r1.0/image_generation/pggan/pggan)** for an example with multiple models and optimizers.

<a id='t05weights'></a>

## Loading model weights

We often need to load the weights of a saved model. Model weights can be loaded by specifying the path of the saved weights using the `weights_path` parameter. Let's use the resnet models created earlier to showcase this.

#### Saving model weights
Here, we create a temporary directory and use FastEstimator backend to save the weights of our previously created resnet50 models:


```python
import os
import tempfile

model_dir = tempfile.mkdtemp()

# TensorFlow
fe.backend.save_model(resnet50_tf, save_dir=model_dir, model_name= "resnet50_tf")

# PyTorch
fe.backend.save_model(resnet50_torch, save_dir=model_dir, model_name= "resnet50_torch")
```




    '/var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpl70q0hk6/resnet50_torch.pt'



#### Loading weights for TensorFlow and PyTorch models


```python
# TensorFlow
resnet50_tf = fe.build(model_fn=lambda: tf.keras.applications.ResNet50(weights=None), 
                       optimizer_fn="adam", 
                       weights_path=os.path.join(model_dir, "resnet50_tf.h5"))
```


```python
# PyTorch
resnet50_torch = fe.build(model_fn=lambda: models.resnet50(pretrained=False), 
                          optimizer_fn="adam", 
                          weights_path=os.path.join(model_dir, "resnet50_torch.pt"))
```

<a id='t05name'></a>

## Specifying a Model Name

The name of a model can be specified using the `model_name` parameter. The name of the model is helpful in distinguishing models when multiple are present.


```python
model = fe.build(model_fn=LeNet, optimizer_fn="adam", model_name="LeNet")
print("Model Name: ", model.model_name)
```

    Model Name:  LeNet


If a model function returns multiple models, a list of model_names can be given. See the **[pggan apphub](examples/r1.0/image_generation/pggan/pggan)** for an illustration with multiple models and model names.

<a id='t05apphub'></a>

## Apphub Examples
You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [PG-GAN](examples/r1.0/image_generation/pggan/pggan)
* [Uncertainty Weighted Loss](examples/r1.0/multi_task_learning/uncertainty_weighted_loss/uncertainty_loss)
