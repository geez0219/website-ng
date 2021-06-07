# Advanced Tutorial 4: Trace

## Overview
In this tutorial, we will discuss:
* [Customizing Traces](tutorials/r1.2/advanced/t04_trace/#ta04customize)
    * [Example](tutorials/r1.2/advanced/t04_trace/#ta04example)
* [More About Traces](tutorials/r1.2/advanced/t04_trace/#ta04more)
    * [Inputs, Outputs, and Mode](tutorials/r1.2/advanced/t04_trace/#ta04iom)
    * [Data](tutorials/r1.2/advanced/t04_trace/#ta04data)
    * [System](tutorials/r1.2/advanced/t04_trace/#ta04system)
* [Trace Communication](tutorials/r1.2/advanced/t04_trace/#ta04communication)
* [Other Trace Usages](tutorials/r1.2/advanced/t04_trace/#ta04other)
    * [Debugging/Monitoring](tutorials/r1.2/advanced/t04_trace/#ta04debug)
* [Related Apphub Examples](tutorials/r1.2/advanced/t04_trace/#ta04apphub)

Let's create a function to generate a pipeline, model and network to be used for the tutorial:


```python
import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp


def get_pipeline_model_network(model_name="LeNet", batch_size=32):
    train_data, eval_data = mnist.load_data()
    test_data = eval_data.split(0.5)
    
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           test_data=test_data,
                           batch_size=batch_size,
                           ops=[ExpandDims(inputs="x", outputs="x"), 
                                Minmax(inputs="x", outputs="x")])

    model = fe.build(model_fn=LeNet, optimizer_fn="adam", model_name=model_name)

    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    return pipeline, model, network
```

<a id='ta04customize'></a>

## Customizing Traces
In [Beginner Tutorial 7](tutorials/r1.2/beginner/t07_estimator), we talked about the basic concept and structure of `Traces` and used a few `Traces` provided by FastEstimator. We can also customize a Trace to suit our needs. Let's look at an example of a custom trace implementation:

<a id='ta04example'></a>

### Example
We can utilize traces to calculate any custom metric needed for monitoring or controlling training. Below, we implement a trace for calculating the F-beta score of our model.


```python
from fastestimator.util import to_number
from fastestimator.trace import Trace
from sklearn.metrics import fbeta_score
import numpy as np

class FBetaScore(Trace):
    def __init__(self, true_key, pred_key, beta=2, output_name="f_beta_score", mode=["eval", "test"]):
        super().__init__(inputs=(true_key, pred_key), outputs=output_name, mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.beta = beta
        self.y_true = []
        self.y_pred = []
        
    def on_epoch_begin(self, data):
        self.y_true = []
        self.y_pred = []
        
    def on_batch_end(self, data):
        y_true, y_pred = to_number(data[self.true_key]), to_number(data[self.pred_key])
        y_pred = np.argmax(y_pred, axis=-1)
        self.y_pred.extend(y_pred.ravel())
        self.y_true.extend(y_true.ravel())
        
    def on_epoch_end(self, data):
        score = fbeta_score(self.y_true, self.y_pred, beta=self.beta, average="weighted")
        data.write_with_log(self.outputs[0], score)
```

Now let's calculate the f2-score using our custom `Trace`. f2-score gives more importance to recall.


```python
pipeline, model, network = get_pipeline_model_network()

traces = FBetaScore(true_key="y", pred_key="y_pred", beta=2, output_name="f2_score", mode="eval")
estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=4, traces=traces, log_steps=1000)

estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 1000; 
    FastEstimator-Train: step: 1; ce: 2.3083596; 
    FastEstimator-Train: step: 1000; ce: 0.16284753; steps/sec: 656.26; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 3.55 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.035797507; f2_score: 0.9885909522565743; 
    FastEstimator-Train: step: 2000; ce: 0.020546585; steps/sec: 615.78; 
    FastEstimator-Train: step: 3000; ce: 0.0059753414; steps/sec: 713.25; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 2.69 sec; 
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.03689827; f2_score: 0.9877924021686296; 
    FastEstimator-Train: step: 4000; ce: 0.02098944; steps/sec: 680.01; 
    FastEstimator-Train: step: 5000; ce: 0.22268356; steps/sec: 741.56; 
    FastEstimator-Train: step: 5625; epoch: 3; epoch_time: 2.65 sec; 
    FastEstimator-Eval: step: 5625; epoch: 3; ce: 0.032033153; f2_score: 0.9901934586365465; 
    FastEstimator-Train: step: 6000; ce: 0.0055854702; steps/sec: 677.84; 
    FastEstimator-Train: step: 7000; ce: 0.0013257915; steps/sec: 679.31; 
    FastEstimator-Train: step: 7500; epoch: 4; epoch_time: 2.8 sec; 
    FastEstimator-Eval: step: 7500; epoch: 4; ce: 0.029642625; f2_score: 0.9913968204671144; 
    FastEstimator-Finish: step: 7500; total_time: 17.99 sec; LeNet_lr: 0.001; 


<a id='ta04more'></a>

## More About Traces
As we have now seen a custom Trace implementaion, let's delve deeper into the structure of `Traces`.

<a id='ta04iom'></a>

### Inputs, Outputs, and Mode
These Trace arguments are similar to the Operator. To recap, the keys from the data dictionary which are required by the Trace can be specified using the `inputs` argument. The `outputs` argument is used to specify the keys which the Trace wants to write into the system buffer. Unlike with Ops, the Trace `inputs` and `outputs` are essentially on an honor system. FastEstimator will not check whether a Trace is really only reading values listed in its `inputs` and writing values listed in its `outputs`. If you are developing a new `Trace` and want your code to work well with the features provided by FastEstimator, it is important to use these fields correctly. The `mode` argument is used to specify the mode(s) for trace execution as with `Ops`. 

<a id='ta04data'></a>

### Data
Through its data argument, Trace has access to the current data dictionary. You can use any keys which the Trace declared as its `inputs` to access information from the data dictionary. You can write the outputs into the `Data` dictionary with or without logging using the `write_with_log` and `write_without_log` methods respectively.

<a id='ta04system'></a>

### System

Traces have access to the current `System` instance which has information about the `Network` and training process. The information contained in `System` is listed below:
* global_step
* num_devices
* log_steps
* total_epochs
* epoch_idx
* batch_idx
* stop_training
* network
* max_train_steps_per_epoch
* max_eval_steps_per_epoch
* summary
* experiment_time

We will showcase `System` usage in the [other trace usages](tutorials/r1.2/advanced/t04_trace/#ta04other) section of this tutorial. 

<a id='ta04communication'></a>

## Trace Communication
We can have multiple traces in a network where the output of one trace is utilized as an input for another, as depicted below: 

<img src=assets/branches/r1.2/tutorial/resources/t04_advanced_trace_communication.png alt="drawing" width="500"/>

Let's see an example where we utilize the outputs of the `Precision` and `Recall` `Traces` to generate f1-score:


```python
from fastestimator.trace.metric import Precision, Recall

class CustomF1Score(Trace):
    def __init__(self, precision_key, recall_key, mode=["eval", "test"], output_name="f1_score"):
        super().__init__(inputs=(precision_key, recall_key), outputs=output_name, mode=mode)
        self.precision_key = precision_key
        self.recall_key = recall_key
        
    def on_epoch_end(self, data):
        precision = data[self.precision_key]
        recall = data[self.recall_key]
        score = 2*(precision*recall)/(precision+recall)
        data.write_with_log(self.outputs[0], score)
        

pipeline, model, network = get_pipeline_model_network()

traces = [
    Precision(true_key="y", pred_key="y_pred", mode=["eval", "test"], output_name="precision"),
    Recall(true_key="y", pred_key="y_pred", mode=["eval", "test"], output_name="recall"),
    CustomF1Score(precision_key="precision", recall_key="recall", mode=["eval", "test"], output_name="f1_score")
]
estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=2, traces=traces, log_steps=1000)
```


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 1000; 
    FastEstimator-Train: step: 1; ce: 2.305337; 
    FastEstimator-Train: step: 1000; ce: 0.024452677; steps/sec: 734.32; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 2.76 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.0569705; 
    precision:
    [0.97585513,0.98211091,0.9752381 ,0.98080614,0.99562363,0.96210526,
     1.        ,0.98137803,1.        ,0.97504798];
    recall:
    [0.99589322,1.        ,0.99224806,0.99223301,0.98484848,0.9827957 ,
     0.95850622,0.98137803,0.95503212,0.97692308];
    f1_score:
    [0.98577236,0.99097473,0.98366955,0.98648649,0.99020675,0.97234043,
     0.97881356,0.98137803,0.9769989 ,0.97598463];
    FastEstimator-Train: step: 2000; ce: 0.0021102745; steps/sec: 674.01; 
    FastEstimator-Train: step: 3000; ce: 0.0089770565; steps/sec: 688.42; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 2.8 sec; 
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.034781747; 
    precision:
    [0.98780488,0.99097473,0.98843931,0.98841699,0.99349241,0.98908297,
     0.99375   ,0.9905303 ,0.97468354,0.98449612];
    recall:
    [0.99794661,1.        ,0.99418605,0.99417476,0.99134199,0.97419355,
     0.98962656,0.97392924,0.98929336,0.97692308];
    f1_score:
    [0.99284985,0.99546691,0.99130435,0.99128751,0.99241603,0.9815818 ,
     0.99168399,0.98215962,0.98193411,0.98069498];
    FastEstimator-Finish: step: 3750; total_time: 8.76 sec; LeNet_lr: 0.001; 


`Note:` precision, recall, and f1-score are displayed for each class

<a id='ta04other'></a>

## Other Trace Usages 

<a id='ta04debug'></a>

### Debugging/Monitoring
Lets implement a custom trace to monitor a model's predictions. Using this, any discrepancy from the expected behavior can be checked and the relevant corrections can be made: 


```python
class MonitorPred(Trace):
    def __init__(self, true_key, pred_key, mode="train"):
        super().__init__(inputs=(true_key, pred_key), mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        
    def on_batch_end(self, data):
        print("Global Step Index: ", self.system.global_step)
        print("Batch Index: ", self.system.batch_idx)
        print("Epoch: ", self.system.epoch_idx)
        print("Batch data has following keys: ", list(data.keys()))
        print("Batch true labels: ", data[self.true_key])
        print("Batch predictictions: ", data[self.pred_key])

pipeline, model, network = get_pipeline_model_network(batch_size=4)

traces = MonitorPred(true_key="y", pred_key="y_pred")
estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=2, traces=traces, max_train_steps_per_epoch=2, log_steps=None)
```


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    Global Step Index:  1
    Batch Index:  1
    Epoch:  1
    Batch data has following keys:  ['y', 'ce', 'x', 'y_pred']
    Batch true labels:  [1 5 8 5]
    Batch predictictions:  [[0.09878654 0.11280762 0.10882236 0.0953772  0.09711165 0.09277759
      0.09783419 0.09401798 0.10111833 0.10134653]
     [0.10425894 0.11605782 0.11004242 0.09267453 0.08793817 0.09537386
      0.10757758 0.08135056 0.09903805 0.10568804]
     [0.1016297  0.11371672 0.10940187 0.09458858 0.09116017 0.09185343
      0.10174091 0.08704273 0.10234813 0.10651773]
     [0.10281158 0.10875763 0.10668261 0.08935054 0.09368025 0.10163527
      0.10554942 0.08158974 0.09799404 0.11194893]]
    Global Step Index:  2
    Batch Index:  2
    Epoch:  1
    Batch data has following keys:  ['y', 'ce', 'x', 'y_pred']
    Batch true labels:  [9 7 0 9]
    Batch predictictions:  [[0.10153595 0.11117928 0.10700106 0.09030598 0.09056976 0.10074646
      0.10491277 0.08370153 0.10058438 0.10946291]
     [0.09943405 0.11675353 0.10615741 0.09357058 0.09498165 0.09680846
      0.09997059 0.08461777 0.09770196 0.11000396]
     [0.10712261 0.11406822 0.10380837 0.09336544 0.08995877 0.09921383
      0.10175668 0.08751085 0.09903854 0.10415668]
     [0.10325367 0.10959569 0.10525871 0.08968467 0.09167413 0.10499243
      0.10512233 0.08271552 0.09867672 0.10902614]]
    Global Step Index:  3
    Batch Index:  1
    Epoch:  2
    Batch data has following keys:  ['y', 'ce', 'x', 'y_pred']
    Batch true labels:  [4 9 5 0]
    Batch predictictions:  [[0.10507825 0.10794099 0.10248892 0.08767187 0.08906174 0.10877317
      0.10675651 0.08316758 0.09733932 0.11172164]
     [0.10452065 0.10935836 0.10143676 0.08643056 0.08772491 0.11231022
      0.10028692 0.08151487 0.09872114 0.11769552]
     [0.10281294 0.11222194 0.1011567  0.08917599 0.093499   0.10987655
      0.10295148 0.08328241 0.09753096 0.10749206]
     [0.11502377 0.10897078 0.10094845 0.08484171 0.08951931 0.10733136
      0.09949591 0.08294778 0.09814924 0.11277179]]
    Global Step Index:  4
    Batch Index:  2
    Epoch:  2
    Batch data has following keys:  ['y', 'ce', 'x', 'y_pred']
    Batch true labels:  [2 9 5 9]
    Batch predictictions:  [[0.10447924 0.11029453 0.09903328 0.08642756 0.09253392 0.11049397
      0.10054693 0.08330047 0.09570859 0.11718156]
     [0.10390399 0.11127824 0.10138535 0.08615676 0.09266223 0.11076459
      0.10240171 0.08131735 0.09794777 0.11218196]
     [0.10628477 0.10850214 0.09937814 0.08383881 0.0902461  0.11622549
      0.103737   0.07806063 0.09677587 0.11695106]
     [0.10669366 0.10886899 0.09865166 0.08427355 0.0894412  0.117375
      0.10394516 0.07848874 0.09449891 0.11776313]]


As you can see, we can visualize information like the global step, batch number, epoch, keys in the data dictionary, true labels, and predictions at batch level using our `Trace`.

<a id='ta04apphub'></a>

## Apphub Examples
You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [CIFAR10](examples/r1.2/image_classification/cifar10_fast/cifar10_fast)
