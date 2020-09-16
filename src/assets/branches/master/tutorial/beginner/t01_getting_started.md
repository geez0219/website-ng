# Tutorial 1: Getting Started

## Overview
Welcome to FastEstimator! In this tutorial we are going to cover:
* [The three main APIs of FastEstimator: `Pipeline`, `Network`, `Estimator`](./tutorials/beginner/t01_getting_started#t01ThreeMain)
* [An image classification example](./tutorials/beginner/t01_getting_started#t01ImageClassification)
    * [Pipeline](./tutorials/beginner/t01_getting_started#t01Pipeline)
    * [Network](./tutorials/beginner/t01_getting_started#t01Network)
    * [Estimator](./tutorials/beginner/t01_getting_started#t01Estimator)
    * [Training](./tutorials/beginner/t01_getting_started#t01Training)
    * [Inferencing](./tutorials/beginner/t01_getting_started#t01Inferencing)
* [Related Apphub Examples](./tutorials/beginner/t01_getting_started#t01Apphub)

<a id='t01ThreeMain'></a>

## Three main APIs
All deep learning training workﬂows involve the following three essential components, each mapping to a critical API in FastEstimator.

* **Data pipeline**: extracts data from disk/RAM, performs transformations. ->  `fe.Pipeline`


* **Network**: performs trainable and differentiable operations. ->  `fe.Network`


* **Training loop**: combines the data pipeline and network in an iterative process. ->  `fe.Estimator`

<BR>
<BR>
Any deep learning task can be constructed by following the 3 main steps:
<img src="assets/branches/master/tutorial/../resources/t01_api.png" alt="drawing" width="700"/>

<a id='t01ImageClassification'></a>

## Image Classification Example

<a id='t01Pipeline'></a>

### Step 1 - Pipeline
We use FastEstimator dataset API to load the MNIST dataset. Please check out [tutorial 2](https://github.com/fastestimator/fastestimator/tree/master/tutorials/beginner/t02_dataset) for more details about the dataset API. In this case our data preprocessing involves: 
1. Expand image dimension from (28,28) to (28, 28, 1) for convenience during convolution operations.
2. Rescale pixel values from [0, 255] to [0, 1].

Please check out [tutorial 3](https://github.com/fastestimator/fastestimator/tree/master/tutorials/beginner/t03_operator) for details about `Operator` and [tutorial 4](https://github.com/fastestimator/fastestimator/tree/master/tutorials/beginner/t04_pipeline) for `Pipeline`.


```python
import fastestimator as fe
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax

train_data, eval_data = mnist.load_data()

pipeline = fe.Pipeline(train_data=train_data,
                       eval_data=eval_data,
                       batch_size=32,
                       ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])
```

<a id='t01Network'></a>

### Step 2 - Network

The model definition can be either from `tf.keras.Model` or `torch.nn.Module`, for more info about network definitions, check out [tutorial 5](https://github.com/fastestimator/fastestimator/tree/master/tutorials/beginner/t05_model). The differentiable operations during training are listed as follows:

1. Feed the preprocessed images to the network and get prediction scores.
2. Calculate `CrossEntropy` (loss) between prediction scores and ground truth.
3. Update the model by minimizing `CrossEntropy`.

For more info about `Network` and its operators, check out [tutorial 6](https://github.com/fastestimator/fastestimator/tree/master/tutorials/beginner/t06_network).


```python
from fastestimator.architecture.tensorflow import LeNet
# from fastestimator.architecture.pytorch import LeNet  # One can also use a pytorch model

from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp

model = fe.build(model_fn=LeNet, optimizer_fn="adam")

network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce") 
    ])
```

<a id='t01Estimator'></a>

### Step 3 - Estimator
We define the `Estimator` to connect the `Network` to the `Pipeline`, and compute accuracy as a validation metric. Please see [tutorial 7](https://github.com/fastestimator/fastestimator/tree/master/tutorials/beginner/t07_estimator) for more about `Estimator` and `Traces`.


```python
from fastestimator.trace.metric import Accuracy
from fastestimator.trace.io import BestModelSaver
import tempfile

traces = [Accuracy(true_key="y", pred_key="y_pred"),
          BestModelSaver(model=model, save_dir=tempfile.mkdtemp(), metric="accuracy", save_best_mode="max")]

estimator = fe.Estimator(pipeline=pipeline,
                         network=network,
                         epochs=2,
                         traces=traces)
```

<a id='t01Training'></a>

### Start Training


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 100; 
    FastEstimator-Train: step: 1; ce: 2.325205; 
    FastEstimator-Train: step: 100; ce: 0.37162033; steps/sec: 161.0; 
    FastEstimator-Train: step: 200; ce: 0.24027318; steps/sec: 166.53; 
    FastEstimator-Train: step: 300; ce: 0.042502172; steps/sec: 160.22; 
    FastEstimator-Train: step: 400; ce: 0.08067161; steps/sec: 160.19; 
    FastEstimator-Train: step: 500; ce: 0.0573852; steps/sec: 149.4; 
    FastEstimator-Train: step: 600; ce: 0.0157291; steps/sec: 146.06; 
    FastEstimator-Train: step: 700; ce: 0.21018827; steps/sec: 140.01; 
    FastEstimator-Train: step: 800; ce: 0.008484628; steps/sec: 135.1; 
    FastEstimator-Train: step: 900; ce: 0.02928259; steps/sec: 128.3; 
    FastEstimator-Train: step: 1000; ce: 0.061196238; steps/sec: 126.4; 
    FastEstimator-Train: step: 1100; ce: 0.06762987; steps/sec: 120.72; 
    FastEstimator-Train: step: 1200; ce: 0.0072296523; steps/sec: 118.11; 
    FastEstimator-Train: step: 1300; ce: 0.08244678; steps/sec: 110.16; 
    FastEstimator-Train: step: 1400; ce: 0.07375234; steps/sec: 105.76; 
    FastEstimator-Train: step: 1500; ce: 0.03207487; steps/sec: 104.01; 
    FastEstimator-Train: step: 1600; ce: 0.1325811; steps/sec: 104.97; 
    FastEstimator-Train: step: 1700; ce: 0.2333475; steps/sec: 99.21; 
    FastEstimator-Train: step: 1800; ce: 0.081265345; steps/sec: 101.39; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 17.21 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmplq_y8tyg/model_best_accuracy.h5
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.05035614; accuracy: 0.9828; since_best_accuracy: 0; max_accuracy: 0.9828; 
    FastEstimator-Train: step: 1900; ce: 0.24747448; steps/sec: 100.72; 
    FastEstimator-Train: step: 2000; ce: 0.056484234; steps/sec: 169.42; 
    FastEstimator-Train: step: 2100; ce: 0.1583787; steps/sec: 186.35; 
    FastEstimator-Train: step: 2200; ce: 0.004822081; steps/sec: 179.8; 
    FastEstimator-Train: step: 2300; ce: 0.027388994; steps/sec: 180.22; 
    FastEstimator-Train: step: 2400; ce: 0.017995346; steps/sec: 183.84; 
    FastEstimator-Train: step: 2500; ce: 0.0071977032; steps/sec: 184.27; 
    FastEstimator-Train: step: 2600; ce: 0.034278065; steps/sec: 182.51; 
    FastEstimator-Train: step: 2700; ce: 0.045357186; steps/sec: 181.42; 
    FastEstimator-Train: step: 2800; ce: 0.057187617; steps/sec: 182.88; 
    FastEstimator-Train: step: 2900; ce: 0.04257428; steps/sec: 178.63; 
    FastEstimator-Train: step: 3000; ce: 0.26984444; steps/sec: 167.96; 
    FastEstimator-Train: step: 3100; ce: 0.026010124; steps/sec: 166.83; 
    FastEstimator-Train: step: 3200; ce: 0.03834851; steps/sec: 161.82; 
    FastEstimator-Train: step: 3300; ce: 0.01365272; steps/sec: 166.79; 
    FastEstimator-Train: step: 3400; ce: 0.015053293; steps/sec: 164.75; 
    FastEstimator-Train: step: 3500; ce: 0.0041770767; steps/sec: 163.45; 
    FastEstimator-Train: step: 3600; ce: 0.0006832063; steps/sec: 162.57; 
    FastEstimator-Train: step: 3700; ce: 0.015146113; steps/sec: 158.26; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 11.0 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmplq_y8tyg/model_best_accuracy.h5
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.0408412; accuracy: 0.9875; since_best_accuracy: 0; max_accuracy: 0.9875; 
    FastEstimator-Finish: step: 3750; total_time: 30.16 sec; model_lr: 0.001; 


<a id='t01Inferencing'></a>

### Inferencing
After training, we can do inferencing on new data with `Pipeline.transform` and `Netowork.transform`. Please checkout [tutorial 8](https://github.com/fastestimator/fastestimator/tree/master/tutorials/beginner/t08_mode) for more details. 


```python
import numpy as np

data = eval_data[0]
data = pipeline.transform(data, mode="eval")
data = network.transform(data, mode="eval")

print("Ground truth class is {}".format(data["y"][0]))
print("Predicted class is {}".format(np.argmax(data["y_pred"])))
img = fe.util.ImgData(x=data["x"])
fig = img.paint_figure()
```

    Ground truth class is 7
    Predicted class is 7



![png](assets/branches/master/tutorial/beginner/t01_getting_started_files/t01_getting_started_19_1.png)


<a id='t01Apphub'></a>

## Apphub Examples
You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [MNIST](https://github.com/fastestimator/fastestimator/tree/master/examples/image_classification/mnist)
* [DNN](https://github.com/fastestimator/fastestimator/tree/master/examples/tabular/dnn)
