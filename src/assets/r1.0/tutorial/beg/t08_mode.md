# Tutorial 8: Mode

## Overview
In this tutorial we are going to cover:
* mode concept 
* when the mode be activated
* how to set mode
* code example

## Mode concept
The development cycle of deep learning application usually takes 4 phases: training, evaluation, testing, inference.
FastEstimator provides 4 corresponding modes: `train`, `eval`, `test`, `infer` that allow users to manage each phase independently. Users have the flexibility to construct the network and pipeline in different ways among those modes. 
Only single mode can be active at a time and then the corresponding topology graph will be retrieved and executed.   

## When the modes are activated
* train: `estimator.fit()` being called, during training cycle
* eval: `estimator.fit()` being called, during evaluation cycle
* test: `estimator.test()` being called.
* infer: `pipeline.transform(mode="infer")` or `network.transform(mode="infer")` being called. (The inference part is later covered in **Tutorial 9: Inference**)

## How to set mode
In the previous tutorials we already knew that `Ops` define the workflow of `Network` and `Pipeline` whereas `Trace` control the training process. All `Op` and `Trace` can be specified with one or more modes where users want them to land. Here are all 5 ways to set the modes.

1. **Setting single mode**<br>
  Specify the desired mode as string. <br>
  Ex: Op(mode="train") <br><br>

2. **Setting multiple mode**<br>
  Put all desired modes in a tuple or list as an argument.<br>
  Ex: Trace(mode=["train", "test"]) <br><br>

3. **Setting exception mode**<br>
  Prefix a "!" on a mode, and then all other modes will have this object. <br>
  Ex: Op(mode="!train") <br><br>

4. **Setting all modes**<br>
  Set the mode argument equal to None. <br>
  Ex: Trace(mode=None) <br><br>

5. **Using default mode setting**<br> 
  Not specify anything in mode argument. Different `Op` and `Trace` have different default mode setting. <br>
  Ex: `UpdateOp` -> default mode: train <br>
      `Accuracy` trace -> default mode: eval, test 


## Code example
In order to enhance readers' idea of modes, we are going to show a example code and visualize the topology graph of each mode.


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
                         epochs=2,
                         traces=Accuracy(true_key="y", pred_key="y_pred")) # default mode=[eval, test]
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.


### Train mode
The following figure is the topology graph in "train" mode. It has complete data pipeline including the data augmentation block, `CoarseDropout`. The data source of the pipeline is "train_data". `Accuracy` block will not exist in this mode because the default mode of that trace is "eval" and "test".

<img src="assets/tutorial/../resources/t08_train_mode.PNG" alt="drawing" width="700"/>

### Eval mode
The following figure is the topology graph in "eval" mode. The data augmentation block is missing and the pipeline data source is "eval_data". `Accuracy` block exist in this mode because of its default trace setting.

<img src="assets/tutorial/../resources/t08_eval_mode.PNG" alt="drawing" width="700"/>

### Test mode
Everything of "test" mode is the same as "eval" mode except that the data source of pipeline has switched to "test_data"

<img src="assets/tutorial/../resources/t08_test_mode.PNG" alt="drawing" width="700"/>

### Infer mode
"Infer" mode only has the minimum operations that model inference needs. Data source is not defined yet at this time point because input data will not be passed until calling the inference function. The detail of running model inference is covered in **Tutorial 9: Inference**. 

<img src="assets/tutorial/../resources/t08_infer_mode.PNG" alt="drawing" width="700"/>
