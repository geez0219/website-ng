# MNIST Image Classification Using LeNet

In this tutorial, we are going to walk through the logic in `lenet_mnist.py` shown below and provide step-by-step instructions.


```python
!cat lenet_mnist.py
```

    # Copyright 2019 The FastEstimator Authors. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================
    import tempfile
    
    import numpy as np
    import tensorflow as tf
    
    import fastestimator as fe
    from fastestimator.architecture import LeNet
    from fastestimator.op.tensorop import Minmax, ModelOp, SparseCategoricalCrossentropy
    from fastestimator.trace import Accuracy, ModelSaver
    
    
    def get_estimator(epochs=2, batch_size=32, steps_per_epoch=None, validation_steps=None, model_dir=tempfile.mkdtemp()):
        # step 1. prepare data
        (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
        train_data = {"x": np.expand_dims(x_train, -1), "y": y_train}
        eval_data = {"x": np.expand_dims(x_eval, -1), "y": y_eval}
        data = {"train": train_data, "eval": eval_data}
        pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Minmax(inputs="x", outputs="x"))
    
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
                                 epochs=epochs,
                                 traces=traces,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps)
        return estimator
    
    
    if __name__ == "__main__":
        est = get_estimator()
        est.fit()


## Step 1: Prepare training and evaluation dataset, create FastEstimator `Pipeline`

`Pipeline` can take both data in memory and data in disk. In this example, we are going to use data in memory by loading data with `tf.keras.datasets.mnist`


```python
import tensorflow as tf
(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
print("train image shape is {}".format(x_train.shape))
print("train label shape is {}".format(y_train.shape))
print("eval image shape is {}".format(x_eval.shape))
print("eval label shape is {}".format(y_eval.shape))
```

    train image shape is (60000, 28, 28)
    train label shape is (60000,)
    eval image shape is (10000, 28, 28)
    eval label shape is (10000,)


The convolution layer requires channel dimension (batch, height, width, channel), therefore, we need to expand the training image and evaluation image by one dimension:


```python
import numpy as np
x_train = np.expand_dims(x_train, -1)
x_eval = np.expand_dims(x_eval, -1)
print("train image shape is {}".format(x_train.shape))
print("eval image shape is {}".format(x_eval.shape))
```

    train image shape is (60000, 28, 28, 1)
    eval image shape is (10000, 28, 28, 1)


For in-memory data in `Pipeline`, the data format should be a nested dictionary like: {"mode1": {"feature1": numpy_array, "feature2": numpy_array, ...}, ...}. Each `mode` can be either `train` or `eval`, in our case, we have both `train` and `eval`.  `feature` is the feature name, in our case, we have `x` and `y`.


```python
data = {"train": {"x": x_train, "y": y_train}, "eval": {"x": x_eval, "y": y_eval}}
```


```python
#Parameters
epochs = 2
batch_size = 32
steps_per_epoch = None
validation_steps = None
```

Now we are ready to define `Pipeline`, we want to apply a `Minmax` online preprocessing to the image feature `x` for both training and evaluation:


```python
import fastestimator as fe
from fastestimator.op.tensorop import Minmax
pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Minmax(inputs="x", outputs="x"))
```

## Step 2: Prepare model, create FastEstimator `Network`

First, we have to define the network architecture in `tf.keras.Model` or `tf.keras.Sequential`, for a popular architecture like LeNet, FastEstimator has it implemented already in [fastestimator.architecture.lenet](https://github.com/fastestimator/fastestimator/blob/master/fastestimator/architecture/lenet.py).  After defining the architecture, users are expected to feed the architecture definition and its associated model name, optimizer and loss name (default to be 'loss') to `FEModel`.


```python
from fastestimator.architecture import LeNet
model = fe.build(model_def=LeNet, model_name="lenet", optimizer="adam", loss_name="loss")
```

Now we are ready to define the `Network`: given with a batch data with key `x` and `y`, we have to work our way to `loss` with series of operators.  `ModelOp` is an operator that contains a model.


```python
from fastestimator.op.tensorop import ModelOp, SparseCategoricalCrossentropy
network = fe.Network(ops=[ModelOp(inputs="x", model=model, outputs="y_pred"), 
                          SparseCategoricalCrossentropy(y_pred="y_pred", y_true="y", outputs="loss")])
```

## Step 3: Configure training, create `Estimator`

During the training loop, we want to: 1) measure accuracy for data data 2) save the model with lowest valdiation loss. `Trace` class is used for anything related to training loop, we will need to import the `Accuracy` and `ModelSaver` trace.


```python
import tempfile
from fastestimator.trace import Accuracy, ModelSaver
save_dir = tempfile.mkdtemp()
traces = [Accuracy(true_key="y", pred_key="y_pred", output_name='acc'),
          ModelSaver(model_name="lenet", save_dir=save_dir, save_best=True)]
```

Now we can define the `Estimator` and specify the training configuation:


```python
estimator = fe.Estimator(network=network, 
                         pipeline=pipeline, 
                         epochs=epochs, 
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps,
                         traces=traces)
```

## Start Training


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 0; total_train_steps: 3750; lenet_lr: 0.001; 
    FastEstimator-Train: step: 0; loss: 2.3004544; 
    FastEstimator-Train: step: 100; loss: 0.4547442; examples/sec: 8394.6; progress: 2.7%; 
    FastEstimator-Train: step: 200; loss: 0.2030383; examples/sec: 9686.5; progress: 5.3%; 
    FastEstimator-Train: step: 300; loss: 0.080203; examples/sec: 10258.1; progress: 8.0%; 
    FastEstimator-Train: step: 400; loss: 0.062838; examples/sec: 10273.1; progress: 10.7%; 
    FastEstimator-Train: step: 500; loss: 0.0470703; examples/sec: 10405.2; progress: 13.3%; 
    FastEstimator-Train: step: 600; loss: 0.0467158; examples/sec: 10365.5; progress: 16.0%; 
    FastEstimator-Train: step: 700; loss: 0.1392532; examples/sec: 10417.7; progress: 18.7%; 
    FastEstimator-Train: step: 800; loss: 0.0294086; examples/sec: 10402.9; progress: 21.3%; 
    FastEstimator-Train: step: 900; loss: 0.0149032; examples/sec: 10410.5; progress: 24.0%; 
    FastEstimator-Train: step: 1000; loss: 0.014766; examples/sec: 10409.2; progress: 26.7%; 
    FastEstimator-Train: step: 1100; loss: 0.0175161; examples/sec: 10416.7; progress: 29.3%; 
    FastEstimator-Train: step: 1200; loss: 0.137861; examples/sec: 10400.5; progress: 32.0%; 
    FastEstimator-Train: step: 1300; loss: 0.0222877; examples/sec: 10348.9; progress: 34.7%; 
    FastEstimator-Train: step: 1400; loss: 0.0604208; examples/sec: 10269.0; progress: 37.3%; 
    FastEstimator-Train: step: 1500; loss: 0.0122311; examples/sec: 10336.3; progress: 40.0%; 
    FastEstimator-Train: step: 1600; loss: 0.0271783; examples/sec: 10260.6; progress: 42.7%; 
    FastEstimator-Train: step: 1700; loss: 0.0319622; examples/sec: 10456.0; progress: 45.3%; 
    FastEstimator-Train: step: 1800; loss: 0.2342918; examples/sec: 10406.9; progress: 48.0%; 
    FastEstimator-ModelSaver: Saving model to /tmp/tmpn6p2rxgv/lenet_best_loss.h5
    FastEstimator-Eval: step: 1875; epoch: 0; loss: 0.0433409; min_loss: 0.043340884; since_best_loss: 0; acc: 0.9864783653846154; 
    FastEstimator-Train: step: 1900; loss: 0.0049661; examples/sec: 6542.9; progress: 50.7%; 
    FastEstimator-Train: step: 2000; loss: 0.0115767; examples/sec: 10326.6; progress: 53.3%; 
    FastEstimator-Train: step: 2100; loss: 0.0116386; examples/sec: 10426.1; progress: 56.0%; 
    FastEstimator-Train: step: 2200; loss: 0.054707; examples/sec: 10326.3; progress: 58.7%; 
    FastEstimator-Train: step: 2300; loss: 0.0158905; examples/sec: 10360.3; progress: 61.3%; 
    FastEstimator-Train: step: 2400; loss: 0.0638738; examples/sec: 10149.9; progress: 64.0%; 
    FastEstimator-Train: step: 2500; loss: 0.286301; examples/sec: 10118.9; progress: 66.7%; 
    FastEstimator-Train: step: 2600; loss: 0.0601029; examples/sec: 10180.6; progress: 69.3%; 
    FastEstimator-Train: step: 2700; loss: 0.0307948; examples/sec: 10176.8; progress: 72.0%; 
    FastEstimator-Train: step: 2800; loss: 0.0652389; examples/sec: 10125.9; progress: 74.7%; 
    FastEstimator-Train: step: 2900; loss: 0.0344164; examples/sec: 10080.1; progress: 77.3%; 
    FastEstimator-Train: step: 3000; loss: 0.0116467; examples/sec: 10137.8; progress: 80.0%; 
    FastEstimator-Train: step: 3100; loss: 0.014677; examples/sec: 10032.5; progress: 82.7%; 
    FastEstimator-Train: step: 3200; loss: 0.0106311; examples/sec: 10248.9; progress: 85.3%; 
    FastEstimator-Train: step: 3300; loss: 0.0515876; examples/sec: 10233.6; progress: 88.0%; 
    FastEstimator-Train: step: 3400; loss: 0.0042517; examples/sec: 10226.0; progress: 90.7%; 
    FastEstimator-Train: step: 3500; loss: 0.0082832; examples/sec: 10434.6; progress: 93.3%; 
    FastEstimator-Train: step: 3600; loss: 0.0010491; examples/sec: 10400.1; progress: 96.0%; 
    FastEstimator-Train: step: 3700; loss: 0.0316844; examples/sec: 10317.0; progress: 98.7%; 
    FastEstimator-ModelSaver: Saving model to /tmp/tmpn6p2rxgv/lenet_best_loss.h5
    FastEstimator-Eval: step: 3750; epoch: 1; loss: 0.0369802; min_loss: 0.03698016; since_best_loss: 0; acc: 0.9883814102564102; 
    FastEstimator-Finish: step: 3750; total_time: 15.08 sec; lenet_lr: 0.001; 


## Inferencing

After training, the model is saved to a temporary folder. we can load the model from file and do inferencing on a sample image.


```python
import os
model_path = os.path.join(save_dir, 'lenet_best_loss.h5')
trained_model = tf.keras.models.load_model(model_path, compile=False)
```

Randomly get one image from validation set and compare the ground truth with model prediction:


```python
from fastestimator.xai import show_image

selected_idx = np.random.randint(10000)
print("test image idx {}, ground truth: {}".format(selected_idx, y_eval[selected_idx]))
show_image(x_eval[selected_idx])

test_image = x_eval[selected_idx]
test_image = np.expand_dims(test_image, 0)
prediction_score = trained_model.predict(test_image)
print("model predicted class is {}".format(np.argmax(prediction_score)))
```

    test image idx 565, ground truth: 4
    model predicted class is 4



![png](assets/example/image_classification/lenet_mnist_files/lenet_mnist_29_1.png)



```python

```
