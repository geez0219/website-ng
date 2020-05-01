# Advanced Tutorial 6: Summary

## Overview
In this tutorial, we will discuss the following topics:
* Experiment Logging
* Experiment Summaries
* Log Parsing
* Summary Visualization
* TensorBoard Visualization

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

## Experiment Logging

As you may have noticed if you have used FastEstimator, log messages are printed to the screen during training. If you want to persist these log messages for later records, you can simply pipe them into a file when launching training from the command line, or else just copy and paste the messages from the console into a persistent file on the disk. FastEstimator allows logging to be controlled via arguments passed to the `Estimator` class, as described in the **[Estimator tutorial](https://github.com/fastestimator/fastestimator/blob/master/tutorial/beginner/t07_estimator.ipynb)**. Let's see an example logging every 120 steps:


```python
est = fe.Estimator(pipeline=pipeline, network=network, epochs=1, traces=traces, log_steps=120)
est.fit()
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; model_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.3170264; model_lr: 0.001; 
    FastEstimator-Train: step: 120; ce: 0.25123408; steps/sec: 150.29; model_lr: 0.0009975198; 
    FastEstimator-Train: step: 240; ce: 0.35014245; steps/sec: 152.42; model_lr: 0.000990021; 
    FastEstimator-Train: step: 360; ce: 0.04853468; steps/sec: 150.63; model_lr: 0.0009775789; 
    FastEstimator-Train: step: 480; ce: 0.023025507; steps/sec: 145.32; model_lr: 0.0009603194; 
    FastEstimator-Train: step: 600; ce: 0.3358984; steps/sec: 143.35; model_lr: 0.00093841663; 
    FastEstimator-Train: step: 720; ce: 0.2457319; steps/sec: 133.05; model_lr: 0.00091209175; 
    FastEstimator-Train: step: 840; ce: 0.11616581; steps/sec: 130.7; model_lr: 0.0008816107; 
    FastEstimator-Train: step: 960; ce: 0.18964969; steps/sec: 121.28; model_lr: 0.00084728113; 
    FastEstimator-Train: step: 1080; ce: 0.28008863; steps/sec: 120.56; model_lr: 0.0008094498; 
    FastEstimator-Train: step: 1200; ce: 0.06563216; steps/sec: 111.93; model_lr: 0.0007684987; 
    FastEstimator-Train: step: 1320; ce: 0.03292639; steps/sec: 113.56; model_lr: 0.0007248414; 
    FastEstimator-Train: step: 1440; ce: 0.015047537; steps/sec: 111.73; model_lr: 0.0006789187; 
    FastEstimator-Train: step: 1560; ce: 0.01693014; steps/sec: 108.18; model_lr: 0.00063119427; 
    FastEstimator-Train: step: 1680; ce: 0.011873012; steps/sec: 97.37; model_lr: 0.0005821501; 
    FastEstimator-Train: step: 1800; ce: 0.03793705; steps/sec: 100.05; model_lr: 0.00053228147; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 17.71 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.044970807; min_ce: 0.044970807; since_best: 0; accuracy: 0.9848; 
    FastEstimator-Finish: step: 1875; total_time: 18.34 sec; model_lr: 0.0005009185; 


## Experiment Summaries

Having log messages on the screen can be handy, but what if you want to access these messages within python? Enter the `Summary` class. `Summary` objects contain information about the training over time, and will be automatically generated when the `Estimator` fit() method is invoked with an experiment name: 


```python
est = fe.Estimator(pipeline=pipeline, network=network, epochs=1, traces=traces, log_steps=500)
summary = est.fit("experiment1")
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; model_lr: 0.0005009185; 
    FastEstimator-Train: step: 1; ce: 0.0016992366; model_lr: 0.001; 
    FastEstimator-Train: step: 500; ce: 0.038246006; steps/sec: 165.22; model_lr: 0.000956986; 
    FastEstimator-Train: step: 1000; ce: 0.004683827; steps/sec: 169.73; model_lr: 0.0008350416; 
    FastEstimator-Train: step: 1500; ce: 0.06307138; steps/sec: 171.45; model_lr: 0.0006552519; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 11.18 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.035052985; min_ce: 0.035052985; since_best: 0; accuracy: 0.9882; 
    FastEstimator-Finish: step: 1875; total_time: 11.71 sec; model_lr: 0.0005009185; 


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
                             {'model_lr': {0: array(0.00050092, dtype=float32),
                               1: array(0.001, dtype=float32),
                               500: array(0.00095699, dtype=float32),
                               1000: array(0.00083504, dtype=float32),
                               1500: array(0.00065525, dtype=float32)},
                              'ce': {1: array(0.00169924, dtype=float32),
                               500: array(0.03824601, dtype=float32),
                               1000: array(0.00468383, dtype=float32),
                               1500: array(0.06307138, dtype=float32)},
                              'steps/sec': {500: array(165.22),
                               1000: array(169.73),
                               1500: array(171.45)},
                              'epoch': {1875: 1},
                              'epoch_time': {1875: array('11.18 sec', dtype='<U9')}}),
                 'eval': defaultdict(dict,
                             {'epoch': {1875: 1},
                              'ce': {1875: array(0.03505298, dtype=float32)},
                              'min_ce': {1875: array(0.03505298, dtype=float32)},
                              'since_best': {1875: array(0)},
                              'accuracy': {1875: array(0.9882)},
                              'total_time': {1875: array('11.71 sec', dtype='<U9')},
                              'model_lr': {1875: array(0.00050092, dtype=float32)}})})



The history field can appear a little daunting, but it is simply a dictionary laid out as follows: {mode: {key: {step: value}}}. Once you have invoked the .fit() method with an experiment name, subsequent calls to .test() will add their results into the same summary dictionary:


```python
summary = est.test()
```

    FastEstimator-Test: step: 1875; epoch: 1; accuracy: 0.9916; 



```python
summary.history
```




    defaultdict(<function fastestimator.summary.summary.Summary.__init__.<locals>.<lambda>()>,
                {'train': defaultdict(dict,
                             {'model_lr': {0: array(0.00050092, dtype=float32),
                               1: array(0.001, dtype=float32),
                               500: array(0.00095699, dtype=float32),
                               1000: array(0.00083504, dtype=float32),
                               1500: array(0.00065525, dtype=float32)},
                              'ce': {1: array(0.00169924, dtype=float32),
                               500: array(0.03824601, dtype=float32),
                               1000: array(0.00468383, dtype=float32),
                               1500: array(0.06307138, dtype=float32)},
                              'steps/sec': {500: array(165.22),
                               1000: array(169.73),
                               1500: array(171.45)},
                              'epoch': {1875: 1},
                              'epoch_time': {1875: array('11.18 sec', dtype='<U9')}}),
                 'eval': defaultdict(dict,
                             {'epoch': {1875: 1},
                              'ce': {1875: array(0.03505298, dtype=float32)},
                              'min_ce': {1875: array(0.03505298, dtype=float32)},
                              'since_best': {1875: array(0)},
                              'accuracy': {1875: array(0.9882)},
                              'total_time': {1875: array('11.71 sec', dtype='<U9')},
                              'model_lr': {1875: array(0.00050092, dtype=float32)}}),
                 'test': defaultdict(dict,
                             {'epoch': {1875: 1},
                              'accuracy': {1875: array(0.9916)}})})



Even if an experiment name was not provided during the .fit() call, it may be provided during the .test() call. The resulting summary object will, however, only contain information from the Test mode.

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



## Log Visualization

While seeing log data as numbers can be informative, visualizations of data are often more useful. FastEstimator provides several ways to visualize log data: from python using `Summary` objects or log files, as well as through the command line. 


```python
fe.summary.logs.visualize_logs(experiments=[summary])
```


![png](assets/tutorial/t06_summary_files/t06_summary_19_0.png)


If you are only interested in visualizing a subset of these log values, it is also possible to whitelist or blacklist values via the 'include_metrics' and 'ignore_metrics' arguments respectively:


```python
fe.summary.logs.visualize_logs(experiments=[summary], include_metrics={"accuracy", "ce"})
```


![png](assets/tutorial/t06_summary_files/t06_summary_21_0.png)


It is also possible to compare logs from different experiments, which can be especially useful when fiddling with hyper-parameter values to determine their effects on training:


```python
fe.summary.logs.parse_log_files(file_paths=["../resources/t06a_exp1.txt", "../resources/t06a_exp2.txt"], log_extension=".txt")
```


![png](assets/tutorial/t06_summary_files/t06_summary_23_0.png)


All of the log files within a given directory can also be compared at the same time, either by using the parse_log_dir() method or via the command line as follows: fastestimator logs --extension .txt --smooth 0 ../resources

## TensorBoard

Of course, no modern AI framework would be complete without TensorBoard integration. In FastEstimator, all that is required to achieve TensorBoard integration is to add the TensorBoard `Trace` to the list of traces passed to the `Estimator`:


```python
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
    TensorBoard(log_dir="../outputs/t06a_tboard", weight_histogram_freq="epoch")
]
est = fe.Estimator(pipeline=pipeline, network=network, epochs=3, traces=traces, log_steps=1000)
est.fit()
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Tensorboard: writing logs to /Users/skynet/Development/fastestimator/tutorial/outputs/t06a_tboard/20200424-144233
    FastEstimator-Start: step: 1; model1_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.3219643; model1_lr: 0.001; 
    FastEstimator-Train: step: 1000; ce: 0.008290021; steps/sec: 77.74; model1_lr: 0.0008350416; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 24.81 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.05034715; min_ce: 0.05034715; since_best: 0; accuracy: 0.982; 
    FastEstimator-Train: step: 2000; ce: 0.048896126; steps/sec: 74.24; model1_lr: 0.00044870423; 
    FastEstimator-Train: step: 3000; ce: 0.0021419637; steps/sec: 76.06; model1_lr: 9.664212e-05; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 24.74 sec; 
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.030854492; min_ce: 0.030854492; since_best: 0; accuracy: 0.9892; 
    FastEstimator-Train: step: 4000; ce: 0.010543499; steps/sec: 75.06; model1_lr: 0.0009891716; 
    FastEstimator-Train: step: 5000; ce: 0.015863674; steps/sec: 73.48; model1_lr: 0.0007506123; 
    FastEstimator-Train: step: 5625; epoch: 3; epoch_time: 25.49 sec; 
    FastEstimator-Eval: step: 5625; epoch: 3; ce: 0.033243153; min_ce: 0.030854492; since_best: 1; accuracy: 0.9896; 
    FastEstimator-Finish: step: 5625; total_time: 77.27 sec; model1_lr: 0.0005009185; 


Now let's launch TensorBoard to visualize our logs. Note that this call will prevent any subsequent Jupyter Notebook cells from running until you manually terminate it.


```python
#!tensorboard --reload_multifile=true --logdir ../outputs/t06a_tboard
```

The TensorBoard display should look something like this:
<img src="assets/tutorial/../resources/t06a_tboard1.png" alt="drawing" width="700"/>
<img src="assets/tutorial/../resources/t06a_tboard2.png" alt="drawing" width="700"/>
