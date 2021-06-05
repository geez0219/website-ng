# Advanced Tutorial 6: Summary

## Overview
In this tutorial, we will discuss the following topics:
* [Experiment Logging](./tutorials/r1.2/advanced/t06_summary#ta06logging)
* [Experiment Summaries](./tutorials/r1.2/advanced/t06_summary#ta06summaries)
* [Log Parsing](./tutorials/r1.2/advanced/t06_summary#ta06parsing)
* [Summary Visualization](./tutorials/r1.2/advanced/t06_summary#ta06visualization)
* [Visualizing Repeat Trials](./tutorials/r1.2/advanced/t06_summary#ta06repeat)
* [TensorBoard Visualization](./tutorials/r1.2/advanced/t06_summary#ta06tboard)

## Preliminary Setup

We will first set up a basic MNIST example for the rest of the demonstrations:


```python
import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.metric import Accuracy
from fastestimator.trace.io import TensorBoard

train_data, eval_data = mnist.load_data()
test_data = eval_data.split(0.5)
pipeline = fe.Pipeline(train_data=train_data,
                       eval_data=eval_data,
                       test_data=test_data,
                       batch_size=32,
                       ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

model = fe.build(model_fn=LeNet, optimizer_fn="adam")
network = fe.Network(ops=[
    ModelOp(model=model, inputs="x", outputs="y_pred"),
    CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
    UpdateOp(model=model, loss_name="ce")
])

traces = [
    Accuracy(true_key="y", pred_key="y_pred"),
    LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3))
]
```

<a id='ta06logging'></a>

## Experiment Logging

As you may have noticed if you have used FastEstimator, log messages are printed to the screen during training. If you want to persist these log messages for later records, you can simply pipe them into a file when launching training from the command line, or else just copy and paste the messages from the console into a persistent file on the disk. FastEstimator allows logging to be controlled via arguments passed to the `Estimator` class, as described in the [Beginner Tutorial 7](./tutorials/r1.2/beginner/t07_estimator). Let's see an example logging every 120 steps:


```python
est = fe.Estimator(pipeline=pipeline, network=network, epochs=1, traces=traces, log_steps=120)
est.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 120; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.3110611; model_lr: 0.0009999998;
    FastEstimator-Train: step: 120; ce: 0.21268827; model_lr: 0.000997478; steps/sec: 120.43;
    FastEstimator-Train: step: 240; ce: 0.07162246; model_lr: 0.0009899376; steps/sec: 107.33;
    FastEstimator-Train: step: 360; ce: 0.2769516; model_lr: 0.0009774548; steps/sec: 101.41;
    FastEstimator-Train: step: 480; ce: 0.0580142; model_lr: 0.0009601558; steps/sec: 95.98;
    FastEstimator-Train: step: 600; ce: 0.10032839; model_lr: 0.0009382152; steps/sec: 88.37;
    FastEstimator-Train: step: 720; ce: 0.06508656; model_lr: 0.00091185456; steps/sec: 82.3;
    FastEstimator-Train: step: 840; ce: 0.124659166; model_lr: 0.00088134; steps/sec: 73.85;
    FastEstimator-Train: step: 960; ce: 0.074202396; model_lr: 0.00084697985; steps/sec: 73.63;
    FastEstimator-Train: step: 1080; ce: 0.03753938; model_lr: 0.0008091209; steps/sec: 72.3;
    FastEstimator-Train: step: 1200; ce: 0.007563473; model_lr: 0.0007681455; steps/sec: 65.26;
    FastEstimator-Train: step: 1320; ce: 0.021064557; model_lr: 0.0007244674; steps/sec: 68.19;
    FastEstimator-Train: step: 1440; ce: 0.025434403; model_lr: 0.00067852775; steps/sec: 66.4;
    FastEstimator-Train: step: 1560; ce: 0.06676559; model_lr: 0.0006307903; steps/sec: 68.13;
    FastEstimator-Train: step: 1680; ce: 0.048038457; model_lr: 0.00058173726; steps/sec: 68.68;
    FastEstimator-Train: step: 1800; ce: 0.09327694; model_lr: 0.0005318639; steps/sec: 68.09;
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 24.91 sec;
    FastEstimator-Eval: step: 1875; epoch: 1; accuracy: 0.9862; ce: 0.04588559;
    FastEstimator-Finish: step: 1875; model_lr: 0.0005005; total_time: 25.75 sec;


<a id='ta06summaries'></a>

## Experiment Summaries

Having log messages on the screen can be handy, but what if you want to access these messages within python? Enter the `Summary` class. `Summary` objects contain information about the training over time, and will be automatically generated when the `Estimator` fit() method is invoked with an experiment name: 


```python
est = fe.Estimator(pipeline=pipeline, network=network, epochs=1, traces=traces, log_steps=500)
summary = est.fit("experiment1")
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 500; num_device: 0;
    FastEstimator-Train: step: 1; ce: 0.0058237277; model_lr: 0.0009999998;
    FastEstimator-Train: step: 500; ce: 0.1087436; model_lr: 0.00095681596; steps/sec: 151.12;
    FastEstimator-Train: step: 1000; ce: 0.052608766; model_lr: 0.00083473074; steps/sec: 134.95;
    FastEstimator-Train: step: 1500; ce: 0.018484306; model_lr: 0.000654854; steps/sec: 134.21;
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 13.23 sec;
    FastEstimator-Eval: step: 1875; epoch: 1; accuracy: 0.9912; ce: 0.027249144;
    FastEstimator-Finish: step: 1875; model_lr: 0.0005005; total_time: 13.98 sec;


Lets take a look at what sort of information is contained within our `Summary` object:


```python
summary.name
```




    'experiment1'




```python
summary.history
```




    defaultdict(<function fastestimator.summary.summary.Summary.__init__.<locals>.<lambda>()>,
                {'train': defaultdict(dict,
                             {'logging_interval': {0: array(500)},
                              'num_device': {0: array(0)},
                              'ce': {1: array(0.00582373, dtype=float32),
                               500: array(0.1087436, dtype=float32),
                               1000: array(0.05260877, dtype=float32),
                               1500: array(0.01848431, dtype=float32)},
                              'model_lr': {1: array(0.001, dtype=float32),
                               500: array(0.00095682, dtype=float32),
                               1000: array(0.00083473, dtype=float32),
                               1500: array(0.00065485, dtype=float32),
                               1875: array(0.0005005, dtype=float32)},
                              'steps/sec': {500: array(151.12),
                               1000: array(134.95),
                               1500: array(134.21)},
                              'epoch': {1875: 1},
                              'epoch_time': {1875: array('13.23 sec', dtype='<U9')},
                              'total_time': {1875: array('13.98 sec', dtype='<U9')}}),
                 'eval': defaultdict(dict,
                             {'epoch': {1875: 1},
                              'accuracy': {1875: array(0.9912)},
                              'ce': {1875: array(0.02724914, dtype=float32)}})})



The history field can appear a little daunting, but it is simply a dictionary laid out as follows: {mode: {key: {step: value}}}. Once you have invoked the .fit() method with an experiment name, subsequent calls to .test() will add their results into the same summary dictionary:


```python
summary = est.test()
```

    FastEstimator-Test: step: 1875; epoch: 1; accuracy: 0.9904; ce: 0.026623823;



```python
summary.history
```




    defaultdict(<function fastestimator.summary.summary.Summary.__init__.<locals>.<lambda>()>,
                {'train': defaultdict(dict,
                             {'logging_interval': {0: array(500)},
                              'num_device': {0: array(0)},
                              'ce': {1: array(0.00582373, dtype=float32),
                               500: array(0.1087436, dtype=float32),
                               1000: array(0.05260877, dtype=float32),
                               1500: array(0.01848431, dtype=float32)},
                              'model_lr': {1: array(0.001, dtype=float32),
                               500: array(0.00095682, dtype=float32),
                               1000: array(0.00083473, dtype=float32),
                               1500: array(0.00065485, dtype=float32),
                               1875: array(0.0005005, dtype=float32)},
                              'steps/sec': {500: array(151.12),
                               1000: array(134.95),
                               1500: array(134.21)},
                              'epoch': {1875: 1},
                              'epoch_time': {1875: array('13.23 sec', dtype='<U9')},
                              'total_time': {1875: array('13.98 sec', dtype='<U9')}}),
                 'eval': defaultdict(dict,
                             {'epoch': {1875: 1},
                              'accuracy': {1875: array(0.9912)},
                              'ce': {1875: array(0.02724914, dtype=float32)}}),
                 'test': defaultdict(dict,
                             {'epoch': {1875: 1},
                              'accuracy': {1875: array(0.9904)},
                              'ce': {1875: array(0.02662382, dtype=float32)}})})



Even if an experiment name was not provided during the .fit() call, it may be provided during the .test() call. The resulting summary object will, however, only contain information from the Test mode.

<a id='ta06parsing'></a>

## Log Parsing

Suppose that you have a log file saved to disk, and you want to create an in-memory `Summary` representation of it. This can be done through FastEstimator logging utilities:


```python
summary = fe.summary.logs.parse_log_file(file_path="../resources/t06a_exp1.txt", file_extension=".txt")
```


```python
summary.name
```




    't06a_exp1'




```python
summary.history['eval']
```




    defaultdict(dict,
                {'epoch': {1875: 1.0, 3750: 2.0, 5625: 3.0},
                 'ce': {1875: 0.03284014, 3750: 0.02343675, 5625: 0.02382297},
                 'min_ce': {1875: 0.03284014, 3750: 0.02343675, 5625: 0.02343675},
                 'since_best': {1875: 0.0, 3750: 0.0, 5625: 1.0},
                 'accuracy': {1875: 0.9882, 3750: 0.992, 5625: 0.9922}})



<a id='ta06visualization'></a>

## Log Visualization

While seeing log data as numbers can be informative, visualizations of data are often more useful. FastEstimator provides several ways to visualize log data: from python using `Summary` objects or log files, as well as through the command line. 


```python
fe.summary.logs.visualize_logs(experiments=[summary])
```


    
![png](assets/branches/r1.2/tutorial/advanced/t06_summary_files/t06_summary_23_0.png)
    


If you are only interested in visualizing a subset of these log values, it is also possible to whitelist or blacklist values via the 'include_metrics' and 'ignore_metrics' arguments respectively:


```python
fe.summary.logs.visualize_logs(experiments=[summary], include_metrics={"accuracy", "ce"})
```


    
![png](assets/branches/r1.2/tutorial/advanced/t06_summary_files/t06_summary_25_0.png)
    


It is also possible to compare logs from different experiments, which can be especially useful when fiddling with hyper-parameter values to determine their effects on training:


```python
fe.summary.logs.parse_log_files(file_paths=["../resources/t06a_exp1.txt", "../resources/t06a_exp2.txt"], log_extension=".txt")
```


    
![png](assets/branches/r1.2/tutorial/advanced/t06_summary_files/t06_summary_27_0.png)
    


All of the log files within a given directory can also be compared at the same time, either by using the parse_log_dir() method or via the command line as follows: fastestimator logs --extension .txt --smooth 0 ../resources

<a id='ta06repeat'></a>

## Visualizing Repeat Trials

Suppose you are running some experiments like the ones above to try and decide which of several experimental configurations is best. For example, suppose you are trying to decide between lossA and lossB. You run 5 experiments with each loss in order to account for randomness, and save the logs as lossA_1.txt, lossA_2.txt, lossB_1.txt, etc. You could use the method described above, for example:


```python
fe.summary.logs.parse_log_dir(dir_path='../resources/t06a_logs', smooth_factor=0)
```


    
![png](assets/branches/r1.2/tutorial/advanced/t06_summary_files/t06_summary_31_0.png)
    


While this is certainly an option, it is not very easy to tell at a glance which of lossA or lossB is superior. Let's use log grouping in order to get a cleaner picture:


```python
fe.summary.logs.parse_log_dir(dir_path='../resources/t06a_logs', smooth_factor=0, group_by=r'(.*)_[\d]+\.txt')
```


    
![png](assets/branches/r1.2/tutorial/advanced/t06_summary_files/t06_summary_33_0.png)
    


Now we are displaying the mean values for lossA and lossB, plus or minus their standard deviations over the 5 experiments. This makes it easy to see that lossA results in a better mcc score and calibration error, whereas lossB has slightly faster training, but the speeds are typically within 1 standard deviation so that might be noise. The group_by argument can take any regex pattern, and if you are using it from the command line, you can simply pass `--group_by _n` as a shortcut to get the regex pattern used above.  

<a id='ta06tboard'></a>

## TensorBoard

Of course, no modern AI framework would be complete without TensorBoard integration. In FastEstimator, all that is required to achieve TensorBoard integration is to add the TensorBoard `Trace` to the list of traces passed to the `Estimator`:


```python
import tempfile
log_dir = tempfile.mkdtemp()

pipeline = fe.Pipeline(train_data=train_data,
                       eval_data=eval_data,
                       test_data=test_data,
                       batch_size=32,
                       ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")], num_process=0)
model = fe.build(model_fn=LeNet, optimizer_fn="adam")
network = fe.Network(ops=[
    ModelOp(model=model, inputs="x", outputs="y_pred"),
    CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
    UpdateOp(model=model, loss_name="ce")
])
traces = [
    Accuracy(true_key="y", pred_key="y_pred"),
    LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3)),
    TensorBoard(log_dir=log_dir, weight_histogram_freq="epoch")
]
est = fe.Estimator(pipeline=pipeline, network=network, epochs=3, traces=traces, log_steps=1000)
est.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Tensorboard: writing logs to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpcjuzn7e3/20210210-131203
    FastEstimator-Start: step: 1; logging_interval: 1000; num_device: 0;
    WARNING:tensorflow:5 out of the last 160 calls to <function TFNetwork._forward_step_static at 0x182454510> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    FastEstimator-Train: step: 1; ce: 2.3189225; model1_lr: 0.0009999998;
    FastEstimator-Train: step: 1000; ce: 0.032560546; model1_lr: 0.00083473074; steps/sec: 57.88;
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 32.88 sec;
    FastEstimator-Eval: step: 1875; epoch: 1; accuracy: 0.9884; ce: 0.04527855;
    FastEstimator-Train: step: 2000; ce: 0.04993862; model1_lr: 0.00044828805; steps/sec: 55.96;
    FastEstimator-Train: step: 3000; ce: 0.03779357; model1_lr: 9.639601e-05; steps/sec: 52.63;
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 36.0 sec;
    FastEstimator-Eval: step: 3750; epoch: 2; accuracy: 0.989; ce: 0.031631112;
    FastEstimator-Train: step: 4000; ce: 0.30943048; model1_lr: 0.0009890847; steps/sec: 51.82;
    FastEstimator-Train: step: 5000; ce: 0.018333856; model1_lr: 0.00075025; steps/sec: 50.96;
    FastEstimator-Train: step: 5625; epoch: 3; epoch_time: 37.18 sec;
    FastEstimator-Eval: step: 5625; epoch: 3; accuracy: 0.9902; ce: 0.03236482;
    FastEstimator-Finish: step: 5625; model1_lr: 0.0005005; total_time: 109.39 sec;


Now let's launch TensorBoard to visualize our logs. Note that this call will prevent any subsequent Jupyter Notebook cells from running until you manually terminate it.


```python
#!tensorboard --reload_multifile=true --logdir /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpb_oy2ihe
```

The TensorBoard display should look something like this:

<img src="assets/branches/r1.2/tutorial/resources/t06a_tboard1.png" alt="drawing" width="700"/>

<img src="assets/branches/r1.2/tutorial/resources/t06a_tboard2.png" alt="drawing" width="700"/>
