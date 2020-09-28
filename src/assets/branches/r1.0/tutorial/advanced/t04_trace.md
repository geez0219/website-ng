
# Advanced Tutorial 4: Trace

## Overview
In this tutorial, we will discuss:
* [Customizing Traces](./tutorials/r1.0/advanced/t04_trace#ta04customize)
    * [Example](./tutorials/r1.0/advanced/t04_trace#ta04example)
* [More About Traces](./tutorials/r1.0/advanced/t04_trace#ta04more)
    * [Inputs, Outputs, and Mode](./tutorials/r1.0/advanced/t04_trace#ta04iom)
    * [Data](./tutorials/r1.0/advanced/t04_trace#ta04data)
    * [System](./tutorials/r1.0/advanced/t04_trace#ta04system)
* [Trace Communication](./tutorials/r1.0/advanced/t04_trace#ta04communication)
* [Other Trace Usages](./tutorials/r1.0/advanced/t04_trace#ta04other)
    * [Debugging/Monitoring](./tutorials/r1.0/advanced/t04_trace#ta04debug)
* [Related Apphub Examples](./tutorials/r1.0/advanced/t04_trace#ta04apphub)

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
In [tutorial 7](./tutorials/r1.0/beginner/t07_estimator) in the beginner section, we talked about the basic concept and structure of `Traces` and used a few `Traces` provided by FastEstimator. We can also customize a Trace to suit our needs. Let's look at an example of a custom trace implementation:

<a id='ta04example'></a>

### Example
We can utilize traces to calculate any custom metric needed for monitoring or controlling training. Below, we implement a trace for calculating the F-beta score of our model.


```python
from fastestimator.backend import to_number
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
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 1000; 
    FastEstimator-Train: step: 1; ce: 2.3049126; 
    FastEstimator-Train: step: 1000; ce: 0.18839744; steps/sec: 121.57; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 19.89 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.04401617; f2_score: 0.9853780424409669; 
    FastEstimator-Train: step: 2000; ce: 0.015927518; steps/sec: 95.05; 
    FastEstimator-Train: step: 3000; ce: 0.07206129; steps/sec: 186.86; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 10.6 sec; 
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.04134067; f2_score: 0.9845700637368479; 
    FastEstimator-Train: step: 4000; ce: 0.008171058; steps/sec: 169.0; 
    FastEstimator-Train: step: 5000; ce: 0.0019764265; steps/sec: 180.37; 
    FastEstimator-Train: step: 5625; epoch: 3; epoch_time: 10.88 sec; 
    FastEstimator-Eval: step: 5625; epoch: 3; ce: 0.029307945; f2_score: 0.9900004384152095; 
    FastEstimator-Train: step: 6000; ce: 0.0135234; steps/sec: 167.19; 
    FastEstimator-Train: step: 7000; ce: 0.04989395; steps/sec: 183.41; 
    FastEstimator-Train: step: 7500; epoch: 4; epoch_time: 10.4 sec; 
    FastEstimator-Eval: step: 7500; epoch: 4; ce: 0.032727916; f2_score: 0.9897883746689528; 
    FastEstimator-Finish: step: 7500; total_time: 54.32 sec; LeNet_lr: 0.001; 


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

We will showcase `System` usage in the [other trace usages](./tutorials/r1.0/advanced/t04_trace#ta04other) section of this tutorial. 

<a id='ta04communication'></a>

## Trace Communication
We can have multiple traces in a network where the output of one trace is utilized as an input for another, as depicted below: 

<img src="assets/branches/r1.0/tutorial/../resources/t04_advanced_trace_communication.png" alt="drawing" width="500"/>

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
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 1000; 
    FastEstimator-Train: step: 1; ce: 2.2952752; 
    FastEstimator-Train: step: 1000; ce: 0.1313241; steps/sec: 179.84; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 10.96 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.04599155; 
    precision:
    [0.98878505,0.99165275,0.98351648,0.99017682,0.99798793,0.98454746,
     0.98198198,0.98294243,0.96296296,0.97402597];
    recall:
    [0.99249531,0.98181818,0.98713235,0.99212598,0.98023715,0.98454746,
     0.98866213,0.96848739,0.98526316,0.98039216];
    f1_score:
    [0.9906367 ,0.98671096,0.9853211 ,0.99115044,0.9890329 ,0.98454746,
     0.98531073,0.97566138,0.97398543,0.9771987 ];
    FastEstimator-Train: step: 2000; ce: 0.0038511096; steps/sec: 164.72; 
    FastEstimator-Train: step: 3000; ce: 0.004517486; steps/sec: 161.99; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 12.14 sec; 
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.034655295; 
    precision:
    [0.9906367 ,0.99505766,0.98899083,0.98635478,0.984375  ,0.98675497,
     0.98868778,0.99353448,0.97717842,0.99107143];
    recall:
    [0.99249531,0.99834711,0.99080882,0.99606299,0.99604743,0.98675497,
     0.99092971,0.96848739,0.99157895,0.96732026];
    f1_score:
    [0.99156514,0.99669967,0.98989899,0.99118511,0.99017682,0.98675497,
     0.98980747,0.98085106,0.98432602,0.97905182];
    FastEstimator-Finish: step: 3750; total_time: 24.53 sec; LeNet_lr: 0.001; 


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
    Batch data has following keys:  ['x', 'y', 'y_pred', 'ce']
    Batch true labels:  tf.Tensor([4 6 6 0], shape=(4,), dtype=uint8)
    Batch predictictions:  tf.Tensor(
    [[0.10117384 0.09088749 0.09792296 0.09737834 0.09084693 0.08700039
      0.11264212 0.10984743 0.10661378 0.10568672]
     [0.0952943  0.09128962 0.10272249 0.10368769 0.09144977 0.08363624
      0.1107841  0.11008291 0.10188652 0.10916632]
     [0.10018928 0.08916146 0.10396809 0.10721539 0.0849424  0.08629669
      0.11222021 0.10986723 0.09851621 0.10762308]
     [0.09981812 0.09000086 0.10369569 0.09561141 0.09411818 0.08580256
      0.11238981 0.10954484 0.10357945 0.10543918]], shape=(4, 10), dtype=float32)
    Global Step Index:  2
    Batch Index:  2
    Epoch:  1
    Batch data has following keys:  ['x', 'y', 'y_pred', 'ce']
    Batch true labels:  tf.Tensor([4 4 9 1], shape=(4,), dtype=uint8)
    Batch predictictions:  tf.Tensor(
    [[0.10240942 0.07996594 0.10190804 0.09579862 0.09545476 0.07724807
      0.12645632 0.1047412  0.1043587  0.11165903]
     [0.10151558 0.08773842 0.09836152 0.09669358 0.0958946  0.07577368
      0.12727338 0.10294375 0.10158429 0.11222118]
     [0.10219741 0.08286765 0.10365716 0.09298524 0.09625786 0.06968912
      0.13070971 0.10312404 0.10423445 0.11427741]
     [0.10077347 0.08387047 0.10196234 0.09324285 0.09473021 0.08261613
      0.11878415 0.1059215  0.11001182 0.10808703]], shape=(4, 10), dtype=float32)
    Global Step Index:  3
    Batch Index:  1
    Epoch:  2
    Batch data has following keys:  ['x', 'y', 'y_pred', 'ce']
    Batch true labels:  tf.Tensor([0 7 7 7], shape=(4,), dtype=uint8)
    Batch predictictions:  tf.Tensor(
    [[0.10566284 0.07728784 0.10565729 0.08178721 0.10713114 0.06507431
      0.13530098 0.09833021 0.10452496 0.11924319]
     [0.10526433 0.08540256 0.0971095  0.08443997 0.1094939  0.06850007
      0.12796785 0.09084202 0.10899913 0.12198068]
     [0.10248369 0.0828173  0.10205018 0.0864138  0.10586432 0.07090016
      0.1273839  0.09568971 0.10854369 0.1178532 ]
     [0.10461577 0.08429881 0.09658652 0.08807645 0.10916384 0.07197928
      0.12543353 0.09240671 0.10978852 0.11765066]], shape=(4, 10), dtype=float32)
    Global Step Index:  4
    Batch Index:  2
    Epoch:  2
    Batch data has following keys:  ['x', 'y', 'y_pred', 'ce']
    Batch true labels:  tf.Tensor([0 5 3 7], shape=(4,), dtype=uint8)
    Batch predictictions:  tf.Tensor(
    [[0.09841534 0.0690296  0.10122424 0.07857155 0.11737346 0.05218776
      0.13999611 0.10599035 0.11199971 0.12521197]
     [0.10094637 0.07799206 0.10599674 0.08304708 0.11446269 0.060531
      0.13092558 0.10104699 0.10494769 0.12010376]
     [0.09200194 0.08393346 0.0990442  0.08482413 0.11270893 0.0664842
      0.12764609 0.10573834 0.11171819 0.11590049]
     [0.10079639 0.08117153 0.10319441 0.08249949 0.11676847 0.06465001
      0.12598662 0.10077127 0.10564327 0.11851855]], shape=(4, 10), dtype=float32)


As you can see, we can visualize information like the global step, batch number, epoch, keys in the data dictionary, true labels, and predictions at batch level using our `Trace`.

<a id='ta04apphub'></a>

## Apphub Examples
You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [CIFAR10](./examples/r1.0/image_classification/cifar10_fast)
