# Tutorial 12: Transfer Learning in FastEstimator

Transfer learning is very frequently used in modern deep learning applications as it can greatly improve the performance and reduce training time.  In this tutorial, we will show you how train from existing weights.

Long story short, user only needs to pass the model path in `fe.build` for transfer learning. We will use a simple MNIST classification as example:


```python
import os
import tempfile

import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture import LeNet
from fastestimator.op.tensorop import Minmax, ModelOp, SparseCategoricalCrossentropy
from fastestimator.trace import Accuracy, ModelSaver
```


```python
model_dir = tempfile.mkdtemp()
print("model will be saved to {}".format(model_dir))
```

## Prepare the pretrained-model by training from scratch


```python
#step 1. prepare data
(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
train_data = {"x": np.expand_dims(x_train, -1), "y": y_train}
eval_data = {"x": np.expand_dims(x_eval, -1), "y": y_eval}
data = {"train": train_data, "eval": eval_data}
pipeline = fe.Pipeline(batch_size=32, data=data, ops=Minmax(inputs="x", outputs="x"))

# step 2. prepare model
model = fe.build(model_def=LeNet, model_name="lenet", optimizer="adam", loss_name="loss")

network = fe.Network(ops=[
    ModelOp(inputs="x", model=model, outputs="y_pred"),
    SparseCategoricalCrossentropy(inputs=("y", "y_pred"), outputs="loss")
])

# step 3.prepare estimator
traces = [
    Accuracy(true_key="y", pred_key="y_pred", output_name='acc'),
    ModelSaver(model_name="lenet", save_dir=model_dir, save_best=True)
]
estimator = fe.Estimator(network=network,
                         pipeline=pipeline,
                         epochs=2,
                         traces=traces)
```


```python
estimator.fit()
```

## Training on existing weights
The previous experiment produced a trained model, now we are going to load the model and continue training for two more batch. Note that the model path is used directly in `fe.build`


```python
model_file_path = os.path.join(model_dir, "lenet_best_loss.h5")
print("the model file path is {}".format(model_file_path))
```


```python
pipeline = fe.Pipeline(batch_size=32, data=data, ops=Minmax(inputs="x", outputs="x"))

model = fe.build(model_def=model_file_path, model_name="lenet", optimizer="adam", loss_name="loss")

network = fe.Network(ops=[
    ModelOp(inputs="x", model=model, outputs="y_pred"),
    SparseCategoricalCrossentropy(inputs=("y", "y_pred"), outputs="loss")
])

estimator = fe.Estimator(network=network,
                         pipeline=pipeline,
                         epochs=1,
                         steps_per_epoch=2,
                         traces=Accuracy(true_key="y", pred_key="y_pred", output_name='acc'))
```


```python
estimator.fit()
```

As we can see, when we use a pretrained weight, with only 2 steps of training, the accuracy is already around 99%. 
