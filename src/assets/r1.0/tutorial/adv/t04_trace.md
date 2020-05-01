# Advanced Tutorial 4: Trace

## Overview
In this tutorial, we will talk on:
* **Customizing Trace**
    * Example
* **More about Trace**
    * inputs, outputs and mode
    * data
    * system
* **Trace communication**
* **Other Trace usages**    
    * Debugging/Monitoring

Let's create a function to generate pipeline, model and network to be used for the tutorial.


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

    model = fe.build(model_fn=LeNet, optimizer_fn="adam", model_names=model_name)

    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    return pipeline, model, network
```

## Customizing Trace
In [Tutorial 7](https://github.com/TortoiseHam/fastestimator/blob/tutorials/summary/tutorial/beginner/t07_estimator.ipynb) in the beginner section, we talked about the basic concept and structure of trace and used few Traces from Fastestimator. We can also customize a Trace to suit our needs. Let's look at an example of a custom trace implementation.

### Example
We can utilize traces to calculate any custom metric needed for mintoring or controlling training. Below, we implement a trace for calculating F-beta score of our model.


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

Here we will calculate f2-score. f2-score gives more importance to recall.


```python
pipeline, model, network = get_pipeline_model_network()

traces = FBetaScore(true_key="y", pred_key="y_pred", beta=2, output_name="f2_score", mode="eval")
estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=4, traces=traces, log_steps=1000)

estimator.fit()
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.3143477; 
    FastEstimator-Train: step: 1000; ce: 0.015297858; steps/sec: 254.94; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 10.92 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.043765396; min_ce: 0.043765396; since_best: 0; f2_score: 0.9861824320950051; 
    FastEstimator-Train: step: 2000; ce: 0.04872855; steps/sec: 229.85; 
    FastEstimator-Train: step: 3000; ce: 0.044155814; steps/sec: 237.7; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 7.92 sec; 
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.03520065; min_ce: 0.03520065; since_best: 0; f2_score: 0.9885849350410258; 
    FastEstimator-Train: step: 4000; ce: 0.0040340982; steps/sec: 243.72; 
    FastEstimator-Train: step: 5000; ce: 0.07454432; steps/sec: 260.57; 
    FastEstimator-Train: step: 5625; epoch: 3; epoch_time: 7.45 sec; 
    FastEstimator-Eval: step: 5625; epoch: 3; ce: 0.028074268; min_ce: 0.028074268; since_best: 0; f2_score: 0.9897964154409412; 
    FastEstimator-Train: step: 6000; ce: 0.0027767532; steps/sec: 246.12; 
    FastEstimator-Train: step: 7000; ce: 0.14164165; steps/sec: 261.06; 
    FastEstimator-Train: step: 7500; epoch: 4; epoch_time: 7.4 sec; 
    FastEstimator-Eval: step: 7500; epoch: 4; ce: 0.030896071; min_ce: 0.028074268; since_best: 1; f2_score: 0.9909951775321262; 
    FastEstimator-Finish: step: 7500; total_time: 37.18 sec; LeNet_lr: 0.001; 


## More about Trace
As we have now seen a custom Trace implementaion, let's delve deeper into the structure of Trace.

### Inputs, Outputs and Mode
These Trace arguments are similar to the Operator. To recap, the keys from the data dictionary which are required by the Trace can be specified using the `inputs` argument. The `outputs` argument is used to specify the keys which the Trace wants to write into the system buffer. `mode` is used to specify the mode(s) for trace execution.

### Data
Through the data argument, Trace has access to current data dictionary. You can use the keys passed through the `inputs` argument to access information from the data dictionary. 
We can write the outputs into the `Data` dictionary with or without logging using `write_with_log` and `write_without_log` methods respectively.

### System
Trace has access to the current `System` instance which has information on network and training. The information provided by System is listed below:
* global_step
* num_devices
* log_steps
* total_epochs
* epoch_idx
* batch_idx
* stop_training
* network
* max_steps_per_epoch
* summary
* experiment_time

We will showcase `System` usage in **Other Trace Usage** section in this tutorial. 

## Trace communication
We can have multiple traces in a network where the output of one trace is utilized by the other as depicted below: 

<img src="assets/tutorial/../resources/t04_advanced_trace_communication.png" alt="drawing" width="500"/>

Below, we demonstrate an example where we utilize the outputs of Precision and Recall traces to generate f1-score


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

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.



```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.2954829; 
    FastEstimator-Train: step: 1000; ce: 0.02208437; steps/sec: 261.23; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 7.6 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.04879124; min_ce: 0.04879124; since_best: 0; 
    precision:
    [0.982     ,0.98813559,0.98452611,0.99804305,0.98807157,0.97550111,
     0.98951782,0.98571429,0.96781116,0.99396378];
    recall:
    [0.99392713,0.99828767,0.97884615,0.96958175,0.994     ,0.98871332,
     0.98128898,0.98773006,0.98471616,0.97821782];
    f1_score:
    [0.98792757,0.99318569,0.98167792,0.98360656,0.99102692,0.98206278,
     0.98538622,0.98672114,0.97619048,0.98602794];
    FastEstimator-Train: step: 2000; ce: 0.029476276; steps/sec: 244.28; 
    FastEstimator-Train: step: 3000; ce: 0.0562297; steps/sec: 259.65; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 7.45 sec; 
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.037549634; min_ce: 0.037549634; since_best: 0; 
    precision:
    [0.98403194,0.99145299,0.99803536,0.97769517,0.98217822,0.98866213,
     0.99154334,0.97983871,0.9954955 ,0.98425197];
    recall:
    [0.99797571,0.99315068,0.97692308,1.        ,0.992     ,0.98419865,
     0.97505198,0.99386503,0.9650655 ,0.99009901];
    f1_score:
    [0.99095477,0.99230111,0.98736638,0.9887218 ,0.98706468,0.98642534,
     0.98322851,0.98680203,0.98004435,0.98716683];
    FastEstimator-Finish: step: 3750; total_time: 16.78 sec; LeNet_lr: 0.001; 


`Note:` precision, recall and f1-score are displayed for each class

## Other Trace usages 

### Debugging/Monitoring
Here, we will implement a custom trace to monitor the predictions. Using this, any discrepancy from the expected behavior can be checked and the relevant corrections can be made. 


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
estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=2, traces=traces, max_steps_per_epoch=2, log_steps=None)
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.



```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    Global Step Index:  1
    Batch Index:  1
    Epoch:  1
    Batch data has following keys:  ['ce', 'y_pred', 'y', 'x']
    Batch true labels:  tf.Tensor([1 0 6 1], shape=(4,), dtype=uint8)
    Batch predictictions:  tf.Tensor(
    [[0.10357114 0.10485268 0.1035656  0.09670787 0.089131   0.10154787
      0.1033091  0.10102929 0.09534299 0.10094254]
     [0.09291946 0.09567764 0.11450008 0.0905508  0.08430735 0.09910616
      0.11450168 0.10635708 0.09766228 0.10441744]
     [0.09475411 0.10670866 0.11174129 0.08387841 0.0786593  0.09865108
      0.12429795 0.10309361 0.09057966 0.10763595]
     [0.10064501 0.10197185 0.10699889 0.09279956 0.09250849 0.09957764
      0.10347461 0.1029356  0.0963361  0.10275227]], shape=(4, 10), dtype=float32)
    Global Step Index:  2
    Batch Index:  2
    Epoch:  1
    Batch data has following keys:  ['ce', 'y_pred', 'y', 'x']
    Batch true labels:  tf.Tensor([9 1 5 4], shape=(4,), dtype=uint8)
    Batch predictictions:  tf.Tensor(
    [[0.10539112 0.12841283 0.10276802 0.0906174  0.07294897 0.09887014
      0.10985964 0.09353343 0.08761767 0.10998084]
     [0.10813517 0.12318959 0.09688332 0.09578475 0.07818311 0.10289599
      0.10992745 0.09717274 0.08684725 0.10098062]
     [0.10764633 0.12101298 0.1079504  0.08947217 0.06668799 0.09578747
      0.12210157 0.09181765 0.08330003 0.11422344]
     [0.10115236 0.12633038 0.10354608 0.09306756 0.07296306 0.10216724
      0.11569481 0.09559171 0.08550758 0.10397922]], shape=(4, 10), dtype=float32)
    Global Step Index:  3
    Batch Index:  1
    Epoch:  2
    Batch data has following keys:  ['ce', 'y_pred', 'y', 'x']
    Batch true labels:  tf.Tensor([9 8 2 4], shape=(4,), dtype=uint8)
    Batch predictictions:  tf.Tensor(
    [[0.09989747 0.1293322  0.09944849 0.08648327 0.0692056  0.11160093
      0.12744734 0.08538151 0.07049793 0.12070523]
     [0.10070158 0.13112241 0.09067103 0.09066574 0.07294965 0.11670754
      0.1274325  0.0859609  0.07270757 0.11108113]
     [0.10449941 0.12324855 0.09050508 0.07887118 0.06976768 0.11688479
      0.14250492 0.08487938 0.0745289  0.11431014]
     [0.10436421 0.12404525 0.09513812 0.0817266  0.06942387 0.10788278
      0.13263398 0.08597295 0.07907981 0.1197325 ]], shape=(4, 10), dtype=float32)
    Global Step Index:  4
    Batch Index:  2
    Epoch:  2
    Batch data has following keys:  ['ce', 'y_pred', 'y', 'x']
    Batch true labels:  tf.Tensor([6 5 6 6], shape=(4,), dtype=uint8)
    Batch predictictions:  tf.Tensor(
    [[0.09452584 0.11557856 0.0994264  0.07691345 0.0780458  0.11294395
      0.13567904 0.08279683 0.08053159 0.12355857]
     [0.09919235 0.11813278 0.09881712 0.07796989 0.07886931 0.11273377
      0.12555735 0.08612843 0.07918419 0.12341478]
     [0.09563012 0.11937355 0.09857126 0.07469492 0.07846729 0.11610405
      0.13567077 0.079809   0.07554168 0.12613735]
     [0.09410203 0.1208031  0.09884479 0.08311246 0.08326188 0.11358377
      0.11815131 0.08606358 0.08020364 0.12187345]], shape=(4, 10), dtype=float32)


As you can see, we can visualize information like the Global step, batch number, epoch, keys in the data dictionary, true labels, predictions at batch level etc. using our trace.
