# Advanced Tutorial 6: Summary

## Overview
In this tutorial, we will discuss the following topics:
* [Experiment Logging](tutorials/r1.1/advanced/t06_summary/#ta06logging)
* [Experiment Summaries](tutorials/r1.1/advanced/t06_summary/#ta06summaries)
* [Log Parsing](tutorials/r1.1/advanced/t06_summary/#ta06parsing)
* [Summary Visualization](tutorials/r1.1/advanced/t06_summary/#ta06visualization)
* [TensorBoard Visualization](tutorials/r1.1/advanced/t06_summary/#ta06tboard)

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

As you may have noticed if you have used FastEstimator, log messages are printed to the screen during training. If you want to persist these log messages for later records, you can simply pipe them into a file when launching training from the command line, or else just copy and paste the messages from the console into a persistent file on the disk. FastEstimator allows logging to be controlled via arguments passed to the `Estimator` class, as described in the [Beginner Tutorial 7](tutorials/r1.1/beginner/t07_estimator). Let's see an example logging every 120 steps:


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
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 120; 
    FastEstimator-Train: step: 1; ce: 2.2443972; model_lr: 0.0009999998; 
    FastEstimator-Train: step: 120; ce: 0.12492387; steps/sec: 669.12; model_lr: 0.000997478; 
    FastEstimator-Train: step: 240; ce: 0.048826903; steps/sec: 672.17; model_lr: 0.0009899376; 
    FastEstimator-Train: step: 360; ce: 0.23642284; steps/sec: 725.35; model_lr: 0.0009774548; 
    FastEstimator-Train: step: 480; ce: 0.09294783; steps/sec: 700.77; model_lr: 0.0009601558; 
    FastEstimator-Train: step: 600; ce: 0.028917177; steps/sec: 699.82; model_lr: 0.0009382152; 
    FastEstimator-Train: step: 720; ce: 0.20153509; steps/sec: 684.74; model_lr: 0.00091185456; 
    FastEstimator-Train: step: 840; ce: 0.17263229; steps/sec: 716.73; model_lr: 0.00088134; 
    FastEstimator-Train: step: 960; ce: 0.015276889; steps/sec: 704.36; model_lr: 0.00084697985; 
    FastEstimator-Train: step: 1080; ce: 0.2609227; steps/sec: 717.46; model_lr: 0.0008091209; 
    FastEstimator-Train: step: 1200; ce: 0.087061; steps/sec: 690.11; model_lr: 0.0007681455; 
    FastEstimator-Train: step: 1320; ce: 0.0999055; steps/sec: 725.65; model_lr: 0.0007244674; 
    FastEstimator-Train: step: 1440; ce: 0.0026154623; steps/sec: 682.23; model_lr: 0.00067852775; 
    FastEstimator-Train: step: 1560; ce: 0.0033505233; steps/sec: 725.46; model_lr: 0.0006307903; 
    FastEstimator-Train: step: 1680; ce: 0.03747531; steps/sec: 692.43; model_lr: 0.00058173726; 
    FastEstimator-Train: step: 1800; ce: 0.01739407; steps/sec: 721.32; model_lr: 0.0005318639; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 3.21 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.038839005; accuracy: 0.988; 
    FastEstimator-Finish: step: 1875; total_time: 4.5 sec; model_lr: 0.0005005; 


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
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 500; 
    FastEstimator-Train: step: 1; ce: 0.10593383; model_lr: 0.0009999998; 
    FastEstimator-Train: step: 500; ce: 0.0012721822; steps/sec: 695.15; model_lr: 0.00095681596; 
    FastEstimator-Train: step: 1000; ce: 0.005035706; steps/sec: 708.72; model_lr: 0.00083473074; 
    FastEstimator-Train: step: 1500; ce: 0.0077150357; steps/sec: 711.83; model_lr: 0.000654854; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 2.73 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.037608337; accuracy: 0.9868; 
    FastEstimator-Finish: step: 1875; total_time: 3.96 sec; model_lr: 0.0005005; 


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
                             {'num_device': {0: array(1)},
                              'logging_interval': {0: array(500)},
                              'ce': {1: array(0.10593383, dtype=float32),
                               500: array(0.00127218, dtype=float32),
                               1000: array(0.00503571, dtype=float32),
                               1500: array(0.00771504, dtype=float32)},
                              'model_lr': {1: array(0.001, dtype=float32),
                               500: array(0.00095682, dtype=float32),
                               1000: array(0.00083473, dtype=float32),
                               1500: array(0.00065485, dtype=float32),
                               1875: array(0.0005005, dtype=float32)},
                              'steps/sec': {500: array(695.15),
                               1000: array(708.72),
                               1500: array(711.83)},
                              'epoch': {1875: 1},
                              'epoch_time': {1875: array('2.73 sec', dtype='<U8')},
                              'total_time': {1875: array('3.96 sec', dtype='<U8')}}),
                 'eval': defaultdict(dict,
                             {'epoch': {1875: 1},
                              'ce': {1875: array(0.03760834, dtype=float32)},
                              'accuracy': {1875: array(0.9868)}})})



The history field can appear a little daunting, but it is simply a dictionary laid out as follows: {mode: {key: {step: value}}}. Once you have invoked the .fit() method with an experiment name, subsequent calls to .test() will add their results into the same summary dictionary:


```python
summary = est.test()
```

    FastEstimator-Test: step: 1875; epoch: 1; accuracy: 0.99; 



```python
summary.history
```




    defaultdict(<function fastestimator.summary.summary.Summary.__init__.<locals>.<lambda>()>,
                {'train': defaultdict(dict,
                             {'num_device': {0: array(1)},
                              'logging_interval': {0: array(500)},
                              'ce': {1: array(0.10593383, dtype=float32),
                               500: array(0.00127218, dtype=float32),
                               1000: array(0.00503571, dtype=float32),
                               1500: array(0.00771504, dtype=float32)},
                              'model_lr': {1: array(0.001, dtype=float32),
                               500: array(0.00095682, dtype=float32),
                               1000: array(0.00083473, dtype=float32),
                               1500: array(0.00065485, dtype=float32),
                               1875: array(0.0005005, dtype=float32)},
                              'steps/sec': {500: array(695.15),
                               1000: array(708.72),
                               1500: array(711.83)},
                              'epoch': {1875: 1},
                              'epoch_time': {1875: array('2.73 sec', dtype='<U8')},
                              'total_time': {1875: array('3.96 sec', dtype='<U8')}}),
                 'eval': defaultdict(dict,
                             {'epoch': {1875: 1},
                              'ce': {1875: array(0.03760834, dtype=float32)},
                              'accuracy': {1875: array(0.9868)}}),
                 'test': defaultdict(dict,
                             {'epoch': {1875: 1},
                              'accuracy': {1875: array(0.99)}})})



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


    
![png](assets/branches/r1.1/tutorial/advanced/t06_summary_files/t06_summary_23_0.png)
    


If you are only interested in visualizing a subset of these log values, it is also possible to whitelist or blacklist values via the 'include_metrics' and 'ignore_metrics' arguments respectively:


```python
fe.summary.logs.visualize_logs(experiments=[summary], include_metrics={"accuracy", "ce"})
```


    
![png](assets/branches/r1.1/tutorial/advanced/t06_summary_files/t06_summary_25_0.png)
    


It is also possible to compare logs from different experiments, which can be especially useful when fiddling with hyper-parameter values to determine their effects on training:


```python
fe.summary.logs.parse_log_files(file_paths=["../resources/t06a_exp1.txt", "../resources/t06a_exp2.txt"], log_extension=".txt")
```


    
![png](assets/branches/r1.1/tutorial/advanced/t06_summary_files/t06_summary_27_0.png)
    


All of the log files within a given directory can also be compared at the same time, either by using the parse_log_dir() method or via the command line as follows: fastestimator logs --extension .txt --smooth 0 ../resources

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
    FastEstimator-Tensorboard: writing logs to /tmp/tmppy1edyce/20200929-185115
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 1000; 
    FastEstimator-Train: step: 1; ce: 2.3263385; model2_lr: 0.0009999998; 
    FastEstimator-Train: step: 1000; ce: 0.02360101; steps/sec: 450.57; model2_lr: 0.00083473074; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 4.19 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.062874146; accuracy: 0.9802; 
    FastEstimator-Train: step: 2000; ce: 0.003110849; steps/sec: 466.04; model2_lr: 0.00044828805; 
    FastEstimator-Train: step: 3000; ce: 0.008105779; steps/sec: 466.15; model2_lr: 9.639601e-05; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 4.07 sec; 
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.027781308; accuracy: 0.9904; 
    FastEstimator-Train: step: 4000; ce: 0.01386946; steps/sec: 458.4; model2_lr: 0.0009890847; 
    FastEstimator-Train: step: 5000; ce: 0.020999195; steps/sec: 470.05; model2_lr: 0.00075025; 
    FastEstimator-Train: step: 5625; epoch: 3; epoch_time: 4.06 sec; 
    FastEstimator-Eval: step: 5625; epoch: 3; ce: 0.028465897; accuracy: 0.9902; 
    FastEstimator-Finish: step: 5625; total_time: 13.63 sec; model2_lr: 0.0005005; 


Now let's launch TensorBoard to visualize our logs. Note that this call will prevent any subsequent Jupyter Notebook cells from running until you manually terminate it.


```python
#!tensorboard --reload_multifile=true --logdir /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpb_oy2ihe
```

The TensorBoard display should look something like this:

<img src=assets/branches/r1.1/tutorial/resources/t06a_tboard1.png alt="drawing" width="700"/>

<img src=assets/branches/r1.1/tutorial/resources/t06a_tboard2.png alt="drawing" width="700"/>
