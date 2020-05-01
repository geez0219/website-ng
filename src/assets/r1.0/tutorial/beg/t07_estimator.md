# Tutorial 7: Estimator

## Overview
In this tutorial, we will talk about:
* **Estimator API**:
    * reducing number of training steps per epoch
    * changing logging behavior
    * monitoring intermediate results during training
* **Trace**
    * Concept
    * Structure
    * Usage
* **Model Testing**

`Estimator` is the API that manages everything related to training loop. It combines `Pipeline` and `Network` together and provides users with fine-grain control of the training loop. Before we demonstrate different ways to control the training loop, let's define a template similar to [tutorial 1](linkneeded) and will use pytorch model this time.


```python
import fastestimator as fe
from fastestimator.architecture.pytorch import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
import tempfile

def get_estimator(log_steps=100, monitor_names=None, use_trace=False, max_steps_per_epoch=None, epochs=2):
    # step 1
    train_data, eval_data = mnist.load_data()
    test_data = eval_data.split(0.5)
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           test_data=test_data,
                           batch_size=32,
                           ops=[ExpandDims(inputs="x", outputs="x", axis=0), Minmax(inputs="x", outputs="x")])
    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="adam", model_names="LeNet")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce1"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = None
    if use_trace:
        traces = [Accuracy(true_key="y", pred_key="y_pred"), 
                  BestModelSaver(model=model, save_dir=tempfile.mkdtemp(), metric="accuracy", save_best_mode="max")]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_steps_per_epoch=max_steps_per_epoch,
                             log_steps=log_steps,
                             monitor_names=monitor_names)
    return estimator
```

Then let's start the training using default argument


```python
est = get_estimator()
est.fit()
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.285437; 
    FastEstimator-Train: step: 100; ce: 0.22269681; steps/sec: 30.75; 
    FastEstimator-Train: step: 200; ce: 0.19469196; steps/sec: 37.19; 
    FastEstimator-Train: step: 300; ce: 0.03941389; steps/sec: 42.21; 
    FastEstimator-Train: step: 400; ce: 0.06724152; steps/sec: 42.59; 
    FastEstimator-Train: step: 500; ce: 0.088666774; steps/sec: 43.4; 
    FastEstimator-Train: step: 600; ce: 0.029547397; steps/sec: 42.92; 
    FastEstimator-Train: step: 700; ce: 0.2446182; steps/sec: 42.78; 
    FastEstimator-Train: step: 800; ce: 0.15599772; steps/sec: 42.56; 
    FastEstimator-Train: step: 900; ce: 0.016024368; steps/sec: 42.19; 
    FastEstimator-Train: step: 1000; ce: 0.11564552; steps/sec: 41.01; 
    FastEstimator-Train: step: 1100; ce: 0.0510619; steps/sec: 42.5; 
    FastEstimator-Train: step: 1200; ce: 0.21494551; steps/sec: 42.41; 
    FastEstimator-Train: step: 1300; ce: 0.010847099; steps/sec: 42.57; 
    FastEstimator-Train: step: 1400; ce: 0.10992938; steps/sec: 42.89; 
    FastEstimator-Train: step: 1500; ce: 0.02174776; steps/sec: 42.44; 
    FastEstimator-Train: step: 1600; ce: 0.06658051; steps/sec: 41.99; 
    FastEstimator-Train: step: 1700; ce: 0.03980174; steps/sec: 41.84; 
    FastEstimator-Train: step: 1800; ce: 0.010390399; steps/sec: 42.35; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 45.75 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.0413347; min_ce: 0.0413347; since_best: 0; 
    FastEstimator-Train: step: 1900; ce: 0.015950574; steps/sec: 37.09; 
    FastEstimator-Train: step: 2000; ce: 0.023892395; steps/sec: 40.64; 
    FastEstimator-Train: step: 2100; ce: 0.007204717; steps/sec: 41.76; 
    FastEstimator-Train: step: 2200; ce: 0.068177156; steps/sec: 42.21; 
    FastEstimator-Train: step: 2300; ce: 0.17505729; steps/sec: 40.72; 
    FastEstimator-Train: step: 2400; ce: 0.05181423; steps/sec: 41.98; 
    FastEstimator-Train: step: 2500; ce: 0.061485767; steps/sec: 41.86; 
    FastEstimator-Train: step: 2600; ce: 0.007129285; steps/sec: 41.87; 
    FastEstimator-Train: step: 2700; ce: 0.0020269149; steps/sec: 40.89; 
    FastEstimator-Train: step: 2800; ce: 0.060143325; steps/sec: 41.28; 
    FastEstimator-Train: step: 2900; ce: 0.12084385; steps/sec: 41.79; 
    FastEstimator-Train: step: 3000; ce: 0.008497677; steps/sec: 41.29; 
    FastEstimator-Train: step: 3100; ce: 0.023160372; steps/sec: 40.6; 
    FastEstimator-Train: step: 3200; ce: 0.038032003; steps/sec: 41.02; 
    FastEstimator-Train: step: 3300; ce: 0.014254327; steps/sec: 40.7; 
    FastEstimator-Train: step: 3400; ce: 0.040137023; steps/sec: 41.29; 
    FastEstimator-Train: step: 3500; ce: 0.08643652; steps/sec: 41.91; 
    FastEstimator-Train: step: 3600; ce: 0.003700078; steps/sec: 41.22; 
    FastEstimator-Train: step: 3700; ce: 0.033798058; steps/sec: 41.68; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 45.44 sec; 
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.041513633; min_ce: 0.0413347; since_best: 1; 
    FastEstimator-Finish: step: 3750; total_time: 94.46 sec; LeNet_lr: 0.001; 


## Estimator API
### Reduce the number of steps per epoch
In general, one epoch means one round of the entire training dataset. If evaluation data is available, evaluation happens after every epoch by default. Consider the following two scenarios:

* Training dataset is too large such that evaluation needs to happen multiple times during one epoch.

* Using different training dataset for different epochs, keep the training steps consistent between epoch.

One easy solution to the above scenarios is to reduce the number of training steps per epoch. For example, if we want to train 1200 steps and split them into 4 epochs:


```python
est = get_estimator(max_steps_per_epoch=300, epochs=4)
est.fit()
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.3069198; 
    FastEstimator-Train: step: 100; ce: 0.39666793; steps/sec: 41.63; 
    FastEstimator-Train: step: 200; ce: 0.25483978; steps/sec: 40.64; 
    FastEstimator-Train: step: 300; ce: 0.103529744; steps/sec: 41.0; 
    FastEstimator-Train: step: 300; epoch: 1; epoch_time: 7.36 sec; 
    FastEstimator-Eval: step: 300; epoch: 1; ce: 0.20608786; min_ce: 0.20608786; since_best: 0; 
    FastEstimator-Train: step: 400; ce: 0.065772764; steps/sec: 41.41; 
    FastEstimator-Train: step: 500; ce: 0.2322329; steps/sec: 41.23; 
    FastEstimator-Train: step: 600; ce: 0.15584615; steps/sec: 41.54; 
    FastEstimator-Train: step: 600; epoch: 2; epoch_time: 7.25 sec; 
    FastEstimator-Eval: step: 600; epoch: 2; ce: 0.094110414; min_ce: 0.094110414; since_best: 0; 
    FastEstimator-Train: step: 700; ce: 0.083361395; steps/sec: 38.76; 
    FastEstimator-Train: step: 800; ce: 0.2183959; steps/sec: 35.5; 
    FastEstimator-Train: step: 900; ce: 0.19299057; steps/sec: 38.06; 
    FastEstimator-Train: step: 900; epoch: 3; epoch_time: 8.02 sec; 
    FastEstimator-Eval: step: 900; epoch: 3; ce: 0.082482256; min_ce: 0.082482256; since_best: 0; 
    FastEstimator-Train: step: 1000; ce: 0.22642905; steps/sec: 37.38; 
    FastEstimator-Train: step: 1100; ce: 0.30729812; steps/sec: 35.27; 
    FastEstimator-Train: step: 1200; ce: 0.037637915; steps/sec: 36.07; 
    FastEstimator-Train: step: 1200; epoch: 4; epoch_time: 8.28 sec; 
    FastEstimator-Eval: step: 1200; epoch: 4; ce: 0.11407161; min_ce: 0.082482256; since_best: 1; 
    FastEstimator-Finish: step: 1200; total_time: 37.42 sec; LeNet_lr: 0.001; 


### Change logging behavior
When the number of training epochs is large, the log can become verbose. You can change the logging behavior by choosing one of following options:
* set `log_steps` to `None` if you do not want to see any training log printed.
* set `log_steps` to 0 if you only wish to see evaluation log.
* set `log_steps` to other number if you want training log to be printed in other frequency.

Let's set the `log_steps` to 0:


```python
est = get_estimator(max_steps_per_epoch=300, epochs=4, log_steps=0)
est.fit()
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; LeNet_lr: 0.001; 
    FastEstimator-Eval: step: 300; epoch: 1; ce: 0.189893; min_ce: 0.189893; since_best: 0; 
    FastEstimator-Eval: step: 600; epoch: 2; ce: 0.09446942; min_ce: 0.09446942; since_best: 0; 
    FastEstimator-Eval: step: 900; epoch: 3; ce: 0.09978603; min_ce: 0.09446942; since_best: 1; 
    FastEstimator-Eval: step: 1200; epoch: 4; ce: 0.08089107; min_ce: 0.08089107; since_best: 0; 
    FastEstimator-Finish: step: 1200; total_time: 35.61 sec; LeNet_lr: 0.001; 


### Monitor intermediate results
You might have noticed that in `Network` of `get_estimator`, there is an op: `CrossEntropy(inputs=("y_pred", "y") outputs="ce1")`. However, the `ce1` never shows up in the training log above. This is because we have a smart filtering in the Network that filters out unused variables to reduce the communication between GPU and CPU. On the contrary, `ce` shows up in log because it serves as a loss that updates the model, by default, we add all losses to logger.

But what if we want to see value of `ce1` throughout the training?

Easy, just add `ce1` to `monitor_names` in `Estimator`.


```python
est = get_estimator(max_steps_per_epoch=300, epochs=4, log_steps=150, monitor_names="ce1")
est.fit()
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1; ce1: 2.291547; ce: 2.291547; 
    FastEstimator-Train: step: 150; ce1: 0.45647815; ce: 0.45647815; steps/sec: 42.42; 
    FastEstimator-Train: step: 300; ce1: 0.0731082; ce: 0.0731082; steps/sec: 42.38; 
    FastEstimator-Train: step: 300; epoch: 1; epoch_time: 7.13 sec; 
    FastEstimator-Eval: step: 300; epoch: 1; ce: 0.15827017; ce1: 0.15827017; min_ce: 0.15827017; since_best: 0; 
    FastEstimator-Train: step: 450; ce1: 0.4427644; ce: 0.4427644; steps/sec: 42.11; 
    FastEstimator-Train: step: 600; ce1: 0.12069794; ce: 0.12069794; steps/sec: 41.2; 
    FastEstimator-Train: step: 600; epoch: 2; epoch_time: 7.2 sec; 
    FastEstimator-Eval: step: 600; epoch: 2; ce: 0.09817337; ce1: 0.09817337; min_ce: 0.09817337; since_best: 0; 
    FastEstimator-Train: step: 750; ce1: 0.024283055; ce: 0.024283055; steps/sec: 41.25; 
    FastEstimator-Train: step: 900; ce1: 0.015608522; ce: 0.015608522; steps/sec: 41.31; 
    FastEstimator-Train: step: 900; epoch: 3; epoch_time: 7.27 sec; 
    FastEstimator-Eval: step: 900; epoch: 3; ce: 0.08314435; ce1: 0.08314435; min_ce: 0.08314435; since_best: 0; 
    FastEstimator-Train: step: 1050; ce1: 0.03369446; ce: 0.03369446; steps/sec: 41.13; 
    FastEstimator-Train: step: 1200; ce1: 0.35390818; ce: 0.35390818; steps/sec: 41.43; 
    FastEstimator-Train: step: 1200; epoch: 4; epoch_time: 7.27 sec; 
    FastEstimator-Eval: step: 1200; epoch: 4; ce: 0.06586083; ce1: 0.06586083; min_ce: 0.06586083; since_best: 0; 
    FastEstimator-Finish: step: 1200; total_time: 35.01 sec; LeNet_lr: 0.001; 


As we can see, both `ce` and `ce1` showed up in the log above. And not surprisingly, their values are identical because because they have the same inputs and forward function.

## Trace
### Concept
Now you might ask: changing logging behavior and monitor extra names are all cool, but where is the fine-grained access of training loop? 

The answer is `Trace`.  `Trace` is a module that can offer you access to training stages and allow you "do stuff" with it. Here are some examples of what `Trace` can do:

* print any training data at any training step
* write results to a file during training
* change learning rate based on some loss conditions
* calculate any metrics 
* order you a pizza after training ends
* ...

So what are the different training stages? They are:

* Beginning of training
* Beginning of epoch
* Beginning of batch
* End of batch
* End of epoch
* End of training

<img src="assets/tutorial/../resources/t07_trace_concept.png" alt="drawing" width="500"/>

As we can see from the illustration above, training process is essentially a nested loop of batch loop and epoch loop. In the nested loop, `Trace` places 6 different "road blocks" for you to leverage.


### Structure
If you are familiar with Keras, you will notice that the structure of `Trace` is very similar to the `Callback` in keras.  Despite the similarity on structure, `Trace` has a lot more capabilities and we will talk about it in depth in [advanced tutorial 4](linkneeded).  Implementation-wise, `Trace` is implemented as a python class with structure like this:


```python
class Trace:
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def on_begin(self, data):
        """Runs once at the beginning of training"""

    def on_epoch_begin(self, data):
        """Runs at the beginning of each epoch"""

    def on_batch_begin(self, data):
        """Runs at the beginning of each batch"""

    def on_batch_end(self, data):
        """Runs at the end of each batch"""

    def on_epoch_end(self, data):
        """Runs at the end of each epoch"""

    def on_end(self, data):
        """Runs once at the end training"""
```

Given the structure, users can customize their own functions at different stage and insert them in the training loop. We will leave the customization of `Trace` to advanced tutorial, for now, let's use some pre-built `Traces` from FastEstimator.

During the training loop of the above example, We want 2 things can happen during loop:
1. Save the model weight if the evaluation loss is the best we have seen
2. Calculate the accuracy during evaluation


```python
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy

est = get_estimator(use_trace=True)
est.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.3081949; 
    FastEstimator-Train: step: 100; ce: 0.5279285; steps/sec: 42.07; 
    FastEstimator-Train: step: 200; ce: 0.25854114; steps/sec: 42.21; 
    FastEstimator-Train: step: 300; ce: 0.087560445; steps/sec: 41.99; 
    FastEstimator-Train: step: 400; ce: 0.15637207; steps/sec: 42.33; 
    FastEstimator-Train: step: 500; ce: 0.12036376; steps/sec: 42.21; 
    FastEstimator-Train: step: 600; ce: 0.14624633; steps/sec: 42.14; 
    FastEstimator-Train: step: 700; ce: 0.13878012; steps/sec: 40.68; 
    FastEstimator-Train: step: 800; ce: 0.029607337; steps/sec: 39.21; 
    FastEstimator-Train: step: 900; ce: 0.020351145; steps/sec: 40.05; 
    FastEstimator-Train: step: 1000; ce: 0.13248421; steps/sec: 41.48; 
    FastEstimator-Train: step: 1100; ce: 0.009485307; steps/sec: 41.83; 
    FastEstimator-Train: step: 1200; ce: 0.0721569; steps/sec: 41.6; 
    FastEstimator-Train: step: 1300; ce: 0.008265268; steps/sec: 41.11; 
    FastEstimator-Train: step: 1400; ce: 0.07632062; steps/sec: 41.18; 
    FastEstimator-Train: step: 1500; ce: 0.01529764; steps/sec: 40.85; 
    FastEstimator-Train: step: 1600; ce: 0.039754625; steps/sec: 41.6; 
    FastEstimator-Train: step: 1700; ce: 0.0038167522; steps/sec: 41.2; 
    FastEstimator-Train: step: 1800; ce: 0.021297162; steps/sec: 40.67; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 45.43 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/5g/d_ny7h211cj3zqkzrtq01s480000gn/T/tmpnyam2hfi/LeNet_best_accuracy.pt
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.054049946; min_ce: 0.054049946; since_best: 0; accuracy: 0.9842; 
    FastEstimator-Train: step: 1900; ce: 0.12951142; steps/sec: 40.29; 
    FastEstimator-Train: step: 2000; ce: 0.074941136; steps/sec: 41.04; 
    FastEstimator-Train: step: 2100; ce: 0.06016096; steps/sec: 41.27; 
    FastEstimator-Train: step: 2200; ce: 0.011193704; steps/sec: 41.14; 
    FastEstimator-Train: step: 2300; ce: 0.010099813; steps/sec: 41.69; 
    FastEstimator-Train: step: 2400; ce: 0.021565676; steps/sec: 41.54; 
    FastEstimator-Train: step: 2500; ce: 0.016332593; steps/sec: 40.86; 
    FastEstimator-Train: step: 2600; ce: 0.0018082643; steps/sec: 41.26; 
    FastEstimator-Train: step: 2700; ce: 0.003422577; steps/sec: 41.52; 
    FastEstimator-Train: step: 2800; ce: 0.03522108; steps/sec: 41.95; 
    FastEstimator-Train: step: 2900; ce: 0.023848219; steps/sec: 42.01; 
    FastEstimator-Train: step: 3000; ce: 0.0037921334; steps/sec: 41.25; 
    FastEstimator-Train: step: 3100; ce: 0.061211947; steps/sec: 41.28; 
    FastEstimator-Train: step: 3200; ce: 0.33038557; steps/sec: 41.48; 
    FastEstimator-Train: step: 3300; ce: 0.02813613; steps/sec: 41.57; 
    FastEstimator-Train: step: 3400; ce: 0.05207818; steps/sec: 40.48; 
    FastEstimator-Train: step: 3500; ce: 0.09768469; steps/sec: 41.82; 
    FastEstimator-Train: step: 3600; ce: 0.0021469605; steps/sec: 41.78; 
    FastEstimator-Train: step: 3700; ce: 0.010005553; steps/sec: 41.94; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 45.33 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/5g/d_ny7h211cj3zqkzrtq01s480000gn/T/tmpnyam2hfi/LeNet_best_accuracy.pt
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.03321769; min_ce: 0.03321769; since_best: 0; accuracy: 0.9898; 
    FastEstimator-Finish: step: 3750; total_time: 93.82 sec; LeNet_lr: 0.001; 


As we can see from the log, the model is saved in predefined location and the accuracy is displayed during evaluation.

## Model Testing

Sometimes you have a separate testing dataset other than training and evaluation data. If you want to evalate the model metrics on test data, you can simply call: 


```python
est.test()
```

    FastEstimator-Test: epoch: 2; accuracy: 0.9892; 


Then all test dataset will go through the `Pipeline` and `Network`, and finally execute the traces(in our case, compute accuracy) just like during the training.
