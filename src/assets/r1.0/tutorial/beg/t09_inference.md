# Tutorial 9: Inference
## Overview
In this tutorial we are going to cover:
* Running inference with transform method
    * Pipeline.transform
    * Network.transform

## Running inference with transform method

Running inference means using a trained deep learning model to get the prediction of input data. Users can use `pipeline.transform` and `network.transform` to feed the data forward and get the computed result in any operation node. Here we are going to use an end-to-end example (the same example code of **Tutorial 8: Mode**) of MNIST image classification to demonstrate how to run inference.  

We first train a deep leaning model with the following code.


```python
import fastestimator as fe
from fastestimator.dataset.data import mnist
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax, CoarseDropout
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.architecture.tensorflow import LeNet


train_data, eval_data = mnist.load_data()
test_data = eval_data.split(0.5)
model = fe.build(model_fn=LeNet, optimizer_fn="adam")

pipeline = fe.Pipeline(train_data=train_data,
                       eval_data=eval_data,
                       test_data=test_data,
                       batch_size=32,
                       ops=[ExpandDims(inputs="x", outputs="x"), #default mode=None
                            Minmax(inputs="x", outputs="x_out", mode=None),  
                            CoarseDropout(inputs="x_out", outputs="x_out", mode="train")])

network = fe.Network(ops=[ModelOp(model=model, inputs="x_out", outputs="y_pred"), #default mode=None
                          CrossEntropy(inputs=("y_pred", "y"), outputs="ce", mode="!infer"),
                          UpdateOp(model=model, loss_name="ce", mode="train")])

estimator = fe.Estimator(pipeline=pipeline,
                         network=network,
                         epochs=1,
                         traces=Accuracy(true_key="y", pred_key="y_pred")) # default mode=[eval, test]
estimator.fit()
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; model_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.306116; 
    FastEstimator-Train: step: 100; ce: 1.3739773; steps/sec: 68.23; 
    FastEstimator-Train: step: 200; ce: 1.0571189; steps/sec: 70.64; 
    FastEstimator-Train: step: 300; ce: 1.3136258; steps/sec: 72.51; 
    FastEstimator-Train: step: 400; ce: 1.0577172; steps/sec: 74.09; 
    FastEstimator-Train: step: 500; ce: 1.0502439; steps/sec: 64.83; 
    FastEstimator-Train: step: 600; ce: 1.0095195; steps/sec: 69.61; 
    FastEstimator-Train: step: 700; ce: 0.89524543; steps/sec: 62.38; 
    FastEstimator-Train: step: 800; ce: 0.9588021; steps/sec: 65.85; 
    FastEstimator-Train: step: 900; ce: 0.8637023; steps/sec: 68.89; 
    FastEstimator-Train: step: 1000; ce: 0.8811163; steps/sec: 73.99; 
    FastEstimator-Train: step: 1100; ce: 0.991605; steps/sec: 71.69; 
    FastEstimator-Train: step: 1200; ce: 1.2436669; steps/sec: 69.57; 
    FastEstimator-Train: step: 1300; ce: 0.9296719; steps/sec: 69.97; 
    FastEstimator-Train: step: 1400; ce: 0.8660065; steps/sec: 62.51; 
    FastEstimator-Train: step: 1500; ce: 0.9682411; steps/sec: 61.42; 
    FastEstimator-Train: step: 1600; ce: 0.73518944; steps/sec: 66.97; 
    FastEstimator-Train: step: 1700; ce: 0.70487094; steps/sec: 64.46; 
    FastEstimator-Train: step: 1800; ce: 0.7994304; steps/sec: 65.99; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 31.39 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.17336462; min_ce: 0.17336462; since_best: 0; accuracy: 0.9462; 
    FastEstimator-Finish: step: 1875; total_time: 34.09 sec; model_lr: 0.001; 


We create a customized function for the following showcase purpose. 


```python
import numpy as np
import tensorflow as tf

def print_dict_but_value(data):
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print("{}: ndarray with shape {}".format(key, value.shape))
        
        elif isinstance(value, tf.Tensor):
            print("{}: tf.Tensor with shape {}".format(key, value.shape))
        
        else:
            print("{}: {}".format(key, value))
```

The following graph shows the complete workflow graph (consisting `Pipeline` and `Network`) of "infer" mode. 

<img src="assets/tutorial/../resources/t09_infer_mode.PNG" alt="drawing" width="700"/>

Our goal is to feed the node "x" with input image and get the prediction result from node "y_pred".

### Pipeline.transform
Pipeline object has a `transform` method that run the pipeline graph ("x" to "x_out") when inference data, a dictionary of keys being node names and values being data ({"x":image}), is inserted. The returned output will be the dictionary of computed results of all pipeline nodes in the type of Numpy array. 

<img src="assets/tutorial/../resources/t09_infer_mode2.PNG" alt="drawing" width="700"/>

Here we take eval_data's first image, package it into a dictionary, and then call `pipeline.transform`. 


```python
import copy 

infer_data = {"x": copy.deepcopy(eval_data[0]["x"])}
print_dict_but_value(infer_data)
```

    x: ndarray with shape (28, 28)



```python
infer_data = pipeline.transform(infer_data, mode="infer")
print_dict_but_value(infer_data)
```

    x: ndarray with shape (1, 28, 28, 1)
    x_out: ndarray with shape (1, 28, 28, 1)


### Network.transform

We then use the network object to call `transform` method that run the netowrk graph("x_out" to "y_pred"). Much alike with `pipeline.transform`, it will generate all nodes' data in the `network` with all data in the type of Tensor. The data type depends on the backend of the network. it is `tf.Tensor` with Tensorflow backend and `torch.Tensor` with Pytorch. Please check out **Tutorial 7: Network** for more detail about `Network` backend). 

<img src="assets/tutorial/../resources/t09_infer_mode3.PNG" alt="drawing" width="700"/>


```python
infer_data = network.transform(infer_data, mode="infer")
print_dict_but_value(infer_data)
```

    x: tf.Tensor with shape (1, 28, 28, 1)
    x_out: tf.Tensor with shape (1, 28, 28, 1)
    y_pred: tf.Tensor with shape (1, 10)


Now we can visualize the input image and compare with its prediction class.


```python
import matplotlib.pyplot as plt 
plt.imshow(np.squeeze(infer_data["x"]), cmap="gray")
print("Prediction class is {}".format(np.argmax(infer_data["y_pred"])))
```

    Prediction class is 1



![png](assets/tutorial/t09_inference_files/t09_inference_16_1.png)

