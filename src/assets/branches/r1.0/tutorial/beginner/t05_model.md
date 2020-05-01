# Tutorial 5: Model

## Overview
In this tutorial we will talk about:
* **Instantiating and Compiling the model**
* **Model function**
* **Optimizer function**
* **Loading model weights**
* **Specifying model name**

## Instantiating and Compiling the model

We need to specify two things to instantiate and compile the model:
* model_fn
* optimizer_fn

Model definitions can be implemented in Tensorflow or Pytorch and instantiated by calling <B>`fe.build`</B> which associates the model with specified optimizer and compiles the model.

## Model Function

`model_fn` should be a function/lambda function which returns either a `tf.keras.Model` or `torch.nn.Module`. We can specify the model architecture through following ways in fastestimator:
* Custom model architecture
* Importing model architecture from fastestimator
* Importing pre-trained models/architectures from pytorch or tensorflow

### Custom model architecture
Let's create a custom model in tensorflow and pytorch for demonstration.

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

### Importing model architecture from fastestimator

Below we import a pytorch LeNet architecture from fastestimator. To view a list of all architectures available in fastestimator, go to [Architectures](https://github.com/fastestimator/fastestimator/tree/master/fastestimator/architecture).


```python
from fastestimator.architecture.pytorch import LeNet
# from fastestimator.architecture.tensorflow import LeNet
# one can also use tensorflow model

model = fe.build(model_fn=LeNet, optimizer_fn="adam")
```

### Importing pre-trained models/architectures from pytorch or tensorflow

Below we show how to define a model function using pre-trained resnet model from tensorflow and pytorch respectively. We load the pre-trained models using a lambda function.

#### Pre-trained model from tf.keras.applications 


```python
resnet50_tf = fe.build(model_fn=lambda: tf.keras.applications.ResNet50(weights='imagenet'), optimizer_fn="adam")
```

#### Pre-trained model from torchvision 


```python
from torchvision import models

resnet50_torch = fe.build(model_fn=lambda: models.resnet50(pretrained=True), optimizer_fn="adam")
```

## Optimizer function

`optimizer_fn` can be a string or lambda function.

### Optimizer from string
Specifying string for `optimizer_fn` loads the optimizer with default parameters. 
List of optimizers and their corresponding strings are listed below:
- Adadelta: 'adadelta'
- Adagrad: 'adagrad'
- Adam: 'adam'
- Adamax: 'adamax'
- RMSprop: 'rmsprop'
- SGD: 'sgd'

### Optimizer from function
To specify specific value of learning rate and other parameters, we need to use lambda function to define the optimizer function.


```python
# Tensorflow 
model_tf = fe.build(model_fn=my_model_tf, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))

# Pytorch
model_torch = fe.build(model_fn=my_model_torch, optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=1e-4))
```

If a model function returns multiple models, list of optimizers can be provided. You can go through **[pggan apphub](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan/pggan.ipynb)** for an example with multiple models and optimizers.

## Loading model weights

We often need to load the weights of a saved model. To achieve this, model weights can be loaded by specifying the path of the saved weights using `weights_path` parameter. Let's use resnet models created earlier to showcase this.

#### Saving model weights
Here, we create a temp directory and use fastestimator backend to save the weights of previously created resnet50 models  


```python
import os
import tempfile

model_dir = tempfile.mkdtemp()

# Tensorflow
fe.backend.save_model(resnet50_tf, save_dir=model_dir, model_name= "resnet50_tf")

# Pytorch
fe.backend.save_model(resnet50_torch, save_dir=model_dir, model_name= "resnet50_torch")
```

    FastEstimator-ModelSaver: saved model to /tmp/tmp_e4z9bh_/resnet50_tf.h5
    FastEstimator-ModelSaver: saved model to /tmp/tmp_e4z9bh_/resnet50_torch.pt


#### Loading weights for tensorflow and pytorch models


```python
# Tensorflow
resnet50_tf = fe.build(model_fn=lambda: tf.keras.applications.ResNet50(weights=None), 
                       optimizer_fn="adam", 
                       weights_path=os.path.join(model_dir, "resnet50_tf.h5"))
```

    Loaded model weights from /tmp/tmp_e4z9bh_/resnet50_tf.h5



```python
# Pytorch
resnet50_torch = fe.build(model_fn=lambda: models.resnet50(pretrained=False), 
                          optimizer_fn="adam", 
                          weights_path=os.path.join(model_dir, "resnet50_torch.pt"))
```

    Loaded model weights from /tmp/tmp_e4z9bh_/resnet50_torch.pt


## Specifying model name

Name of the model can be specified using `model_names` parameter. The name of the model is helpful in distinguishing the model in presence of multiple models.


```python
model = fe.build(model_fn=LeNet, optimizer_fn="adam", model_names="LeNet")
print("Model Name: ", model.model_name)
```

    Model Name:  LeNet


If a model function returns multiple models, list of model_names can be given. You can go through **[pggan apphub](https://github.com/fastestimator/fastestimator/blob/master/apphub/image_generation/pggan/pggan.ipynb)** for an illustration with multiple models and model names.
