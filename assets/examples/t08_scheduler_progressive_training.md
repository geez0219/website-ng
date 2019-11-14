# Tutorial 8: Changing hyperparameters during training with Scheduler

Before progressive training emerged, people had to use the same hyperparameters during the whole training. __Progressive training__ is essentially adding a time dimension in hyperparameters to allow any of them to change during the training loop. 

Examples of progressive training use cases:
1. Use a batch size of 32 for the 0th epoch, then use 64 on the 5th epoch.
2. Train with low resolution image (28x28) for the first 3 epochs, then double the resolution (52x52) for another 3 epochs.
3. Train part of the model for the first 10 epochs, then train another part of the model for 10 more epochs.

All of the examples above illustrate __hyperparameter change during the training__. In FastEstimator, `Scheduler` is used to handle these sort of requests. 

## 1) How to use Scheduler:

Scheduler can be used in `Pipeline` and `Network`.  Before using Scheduler, user will need to create a dictionary where the key will be the epoch number, and the value whatever value the user wants to use from that epoch onwards. For example, changing batch size on different epoch:

```python
from fastestimator.schedule import Scheduler

mapping = {0: 32, 2:64, 5: 128}
batchsize_scheduler = Scheduler(epoch_dict=mapping)
```

Then `batchsize_scheduler` can be used directly as batch size in `Pipeline`. Please note that the key in the dictionary indicates the epoch of change, therefore, in the example above, when the total training epoch is 8, the batch size for each epoch is:

* epoch 0, batch size 32
* epoch 1, batch size 32
* epoch 2, batch size 64
* epoch 3, batch size 64
* epoch 4, batch size 64
* epoch 5, batch size 128
* epoch 6, batch size 128
* epoch 7, batch size 128

## 2) Scheduler example:

In the next example, we'll define two image classification models with the same architecture(`model1` and `model2`). We want to train them by the following:

* on epoch 0:  train `model1` with batch size 32, use image resolution 30x30 and Minmax normalization.
* on epoch 1:  train `model2` with batch size 64, use image resolution 32x32 and Minmax normalization.
* on epoch 2:  train `model1` with batch size 128, use image resolution 30x30 and Rescale normalization(multiply by 1/255).

### Step 0- Prepare data


```python
import numpy as np
import tensorflow as tf
import fastestimator as fe

# We load MNIST dataset
(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
train_data = {"x": np.expand_dims(x_train, -1), "y": y_train}
eval_data = {"x": np.expand_dims(x_eval, -1), "y": y_eval}
data = {"train": train_data, "eval": eval_data}
```

### Step 1- Prepare the Pipeline with the Schedulers


```python
from fastestimator.schedule import Scheduler
from fastestimator.op.tensorop import Minmax, Resize, Scale

# We create a scheduler for batch_size with the epochs at which it will change and corresponding values.
batchsize_scheduler = Scheduler({0:32, 1:64, 2:128})

# We create a scheduler for the Resize ops.
resize_scheduler = Scheduler({0: Resize(inputs="x", size=(30, 30), outputs="x"),
                              1: Resize(inputs="x", size=(32, 32), outputs="x"),
                              2: Resize(inputs="x", size=(30, 30), outputs="x")})

# We create a scheduler for the different normalize ops we will want to use.
normalize_scheduler = Scheduler({0: Minmax(inputs="x", outputs="x"),
                                 2: Scale(inputs="x", scalar=1.0/255, outputs="x")})

# In Pipeline, we use the schedulers for batch_size and ops.
pipeline = fe.Pipeline(batch_size=batchsize_scheduler, 
                       data=data, 
                       ops=[resize_scheduler, normalize_scheduler])
```

### Step 2- Prepare Network with the two models and a Scheduler


```python
from fastestimator.architecture import LeNet
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.op.tensorop.loss import SparseCategoricalCrossentropy

# We create two models and build them with their optimizer and loss.
model1 = fe.build(model_def=lambda: LeNet(input_shape=(30,30,1)), model_name="model1", optimizer="adam", loss_name='my_loss')
model2 = fe.build(model_def=lambda: LeNet(input_shape=(32,32,1)), model_name="model2", optimizer="adam", loss_name='my_loss')

# We create a Scheduler to indicate what model we want to train for each epoch.
model_scheduler = Scheduler({0: ModelOp(inputs="x", model=model1, outputs="y_pred"),
                             1: ModelOp(inputs="x", model=model2, outputs="y_pred"),
                             2: ModelOp(inputs="x", model=model1, outputs="y_pred")})

# We summarize the ops in Network, using model_scheduler for ModelOp.
network = fe.Network(ops=[model_scheduler, SparseCategoricalCrossentropy(inputs=("y", "y_pred"), outputs='my_loss')])
```

### Step 3- Build the Estimator and train!


```python
estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=3)
```


```python
estimator.fit()
```
