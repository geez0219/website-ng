# Tutorial 7: Estimator

## Overview
In this tutorial, we will talk about:
* [Estimator API](./tutorials/r1.0/beginner/t07_estimator#t07estimator)
    * [Reducing the number of training steps per epoch](./tutorials/r1.0/beginner/t07_estimator#t07train)
    * [Reducing the number of evaluation steps per epoch](./tutorials/r1.0/beginner/t07_estimator#t07eval)
    * [Changing logging behavior](./tutorials/r1.0/beginner/t07_estimator#t07logging)
    * [Monitoring intermediate results during training](./tutorials/r1.0/beginner/t07_estimator#t07intermediate)
* [Trace](./tutorials/r1.0/beginner/t07_estimator#t07trace)
    * [Concept](./tutorials/r1.0/beginner/t07_estimator#t07concept)
    * [Structure](./tutorials/r1.0/beginner/t07_estimator#t07structure)
    * [Usage](./tutorials/r1.0/beginner/t07_estimator#t07usage)
* [Model Testing](./tutorials/r1.0/beginner/t07_estimator#t07testing)
* [Related Apphub Examples](./tutorials/r1.0/beginner/t07_estimator#t07apphub)

`Estimator` is the API that manages everything related to the training loop. It combines `Pipeline` and `Network` together and provides users with fine-grain control over the training loop. Before we demonstrate different ways to control the training loop let's define a template similar to [Tutorial 1](./tutorials/r1.0/beginner/t01_getting_started), but this time we will use a PyTorch model.


```python
import fastestimator as fe
from fastestimator.architecture.pytorch import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
import tempfile

def get_estimator(log_steps=100, monitor_names=None, use_trace=False, max_train_steps_per_epoch=None, epochs=2):
    # step 1
    train_data, eval_data = mnist.load_data()
    test_data = eval_data.split(0.5)
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           test_data=test_data,
                           batch_size=32,
                           ops=[ExpandDims(inputs="x", outputs="x", axis=0), Minmax(inputs="x", outputs="x")])
    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="adam", model_name="LeNet")
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
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             log_steps=log_steps,
                             monitor_names=monitor_names)
    return estimator
```

Let's train our model using the default `Estimator` arguments:


```python
est = get_estimator()
est.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 100; 
    FastEstimator-Train: step: 1; ce: 2.2985432; 
    FastEstimator-Train: step: 100; ce: 0.33677763; steps/sec: 43.98; 
    FastEstimator-Train: step: 200; ce: 0.3549296; steps/sec: 45.97; 
    FastEstimator-Train: step: 300; ce: 0.17926084; steps/sec: 46.62; 
    FastEstimator-Train: step: 400; ce: 0.32462734; steps/sec: 46.91; 
    FastEstimator-Train: step: 500; ce: 0.05164891; steps/sec: 47.18; 
    FastEstimator-Train: step: 600; ce: 0.0906372; steps/sec: 45.5; 
    FastEstimator-Train: step: 700; ce: 0.46759754; steps/sec: 45.0; 
    FastEstimator-Train: step: 800; ce: 0.025921348; steps/sec: 43.85; 
    FastEstimator-Train: step: 900; ce: 0.21584965; steps/sec: 44.17; 
    FastEstimator-Train: step: 1000; ce: 0.1303818; steps/sec: 44.68; 
    FastEstimator-Train: step: 1100; ce: 0.256935; steps/sec: 43.92; 
    FastEstimator-Train: step: 1200; ce: 0.052581083; steps/sec: 43.21; 
    FastEstimator-Train: step: 1300; ce: 0.030862458; steps/sec: 42.97; 
    FastEstimator-Train: step: 1400; ce: 0.115828656; steps/sec: 42.55; 
    FastEstimator-Train: step: 1500; ce: 0.033370342; steps/sec: 43.89; 
    FastEstimator-Train: step: 1600; ce: 0.0928934; steps/sec: 43.56; 
    FastEstimator-Train: step: 1700; ce: 0.05145497; steps/sec: 43.06; 
    FastEstimator-Train: step: 1800; ce: 0.14278823; steps/sec: 43.23; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 42.33 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.057005133; 
    FastEstimator-Train: step: 1900; ce: 0.08283445; steps/sec: 39.21; 
    FastEstimator-Train: step: 2000; ce: 0.031674776; steps/sec: 46.4; 
    FastEstimator-Train: step: 2100; ce: 0.022434138; steps/sec: 46.2; 
    FastEstimator-Train: step: 2200; ce: 0.0041575576; steps/sec: 46.57; 
    FastEstimator-Train: step: 2300; ce: 0.028007038; steps/sec: 46.55; 
    FastEstimator-Train: step: 2400; ce: 0.11569328; steps/sec: 46.18; 
    FastEstimator-Train: step: 2500; ce: 0.1477213; steps/sec: 46.04; 
    FastEstimator-Train: step: 2600; ce: 0.21895751; steps/sec: 45.41; 
    FastEstimator-Train: step: 2700; ce: 0.008701714; steps/sec: 44.15; 
    FastEstimator-Train: step: 2800; ce: 0.006247335; steps/sec: 42.0; 
    FastEstimator-Train: step: 2900; ce: 0.0016122407; steps/sec: 42.0; 
    FastEstimator-Train: step: 3000; ce: 0.005287632; steps/sec: 41.4; 
    FastEstimator-Train: step: 3100; ce: 0.013425731; steps/sec: 41.41; 
    FastEstimator-Train: step: 3200; ce: 0.00874802; steps/sec: 39.84; 
    FastEstimator-Train: step: 3300; ce: 0.025417497; steps/sec: 40.25; 
    FastEstimator-Train: step: 3400; ce: 0.08027805; steps/sec: 39.33; 
    FastEstimator-Train: step: 3500; ce: 0.020149795; steps/sec: 39.69; 
    FastEstimator-Train: step: 3600; ce: 0.010977306; steps/sec: 39.68; 
    FastEstimator-Train: step: 3700; ce: 0.075040415; steps/sec: 39.68; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 44.2 sec; 
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.04138615; 
    FastEstimator-Finish: step: 3750; total_time: 89.69 sec; LeNet_lr: 0.001; 


<a id='t07estimator'></a>

## Estimator API

<a id='t07train'></a>

### Reduce the number of training steps per epoch
In general, one epoch of training means that every element in the training dataset will be visited exactly one time. If evaluation data is available, evaluation happens after every epoch by default. Consider the following two scenarios:

* The training dataset is very large such that evaluation needs to happen multiple times during one epoch.
* Different training datasets are being used for different epochs, but the number of training steps should be consistent between each epoch.

One easy solution to the above scenarios is to limit the number of training steps per epoch. For example, if we want to train for only 300 steps per epoch, with training lasting for 4 epochs (1200 steps total), we would do the following:


```python
est = get_estimator(max_train_steps_per_epoch=300, epochs=4)
est.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 100; 
    FastEstimator-Train: step: 1; ce: 2.3073506; 
    FastEstimator-Train: step: 100; ce: 0.5364497; steps/sec: 38.56; 
    FastEstimator-Train: step: 200; ce: 0.17832895; steps/sec: 42.4; 
    FastEstimator-Train: step: 300; ce: 0.2198829; steps/sec: 41.62; 
    FastEstimator-Train: step: 300; epoch: 1; epoch_time: 7.42 sec; 
    FastEstimator-Eval: step: 300; epoch: 1; ce: 0.15399536; 
    FastEstimator-Train: step: 400; ce: 0.13039914; steps/sec: 38.44; 
    FastEstimator-Train: step: 500; ce: 0.120313495; steps/sec: 42.95; 
    FastEstimator-Train: step: 600; ce: 0.14686579; steps/sec: 43.12; 
    FastEstimator-Train: step: 600; epoch: 2; epoch_time: 7.25 sec; 
    FastEstimator-Eval: step: 600; epoch: 2; ce: 0.10223439; 
    FastEstimator-Train: step: 700; ce: 0.17189693; steps/sec: 37.89; 
    FastEstimator-Train: step: 800; ce: 0.025620187; steps/sec: 41.49; 
    FastEstimator-Train: step: 900; ce: 0.017038438; steps/sec: 41.58; 
    FastEstimator-Train: step: 900; epoch: 3; epoch_time: 7.46 sec; 
    FastEstimator-Eval: step: 900; epoch: 3; ce: 0.06282204; 
    FastEstimator-Train: step: 1000; ce: 0.038011674; steps/sec: 37.24; 
    FastEstimator-Train: step: 1100; ce: 0.03683513; steps/sec: 42.89; 
    FastEstimator-Train: step: 1200; ce: 0.023527239; steps/sec: 41.78; 
    FastEstimator-Train: step: 1200; epoch: 4; epoch_time: 7.41 sec; 
    FastEstimator-Eval: step: 1200; epoch: 4; ce: 0.079378836; 
    FastEstimator-Finish: step: 1200; total_time: 36.24 sec; LeNet_lr: 0.001; 


<a id='t07eval'></a>

### Reduce the number of evaluation steps per epoch
One may need to reduce the number of evaluation steps for debugging purpose. This can be easily done by setting the `max_eval_steps_per_epoch` argument in `Estimator`.

<a id='t07logging'></a>

### Change logging behavior
When the number of training epochs is large, the log can become verbose. You can change the logging behavior by choosing one of following options:
* set `log_steps` to `None` if you do not want to see any training logs printed.
* set `log_steps` to 0 if you only wish to see the evaluation logs.
* set `log_steps` to some integer 'x' if you want training logs to be printed every 'x' steps.

Let's set the `log_steps` to 0:


```python
est = get_estimator(max_train_steps_per_epoch=300, epochs=4, log_steps=0)
est.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 0; 
    FastEstimator-Eval: step: 300; epoch: 1; ce: 0.15603326; 
    FastEstimator-Eval: step: 600; epoch: 2; ce: 0.09531953; 
    FastEstimator-Eval: step: 900; epoch: 3; ce: 0.06877253; 
    FastEstimator-Eval: step: 1200; epoch: 4; ce: 0.05356282; 
    FastEstimator-Finish: step: 1200; total_time: 36.81 sec; LeNet_lr: 0.001; 


<a id='t07intermediate'></a>

### Monitor intermediate results
You might have noticed that in our example `Network` there is an op: `CrossEntropy(inputs=("y_pred", "y") outputs="ce1")`. However, the `ce1` never shows up in the training log above. This is because FastEstimator identifies and filters out unused variables to reduce unnecessary communication between the GPU and CPU. On the contrary, `ce` shows up in the log because by default we log all loss values that are used to update models.

But what if we want to see the value of `ce1` throughout training?

Easy: just add `ce1` to `monitor_names` in `Estimator`.


```python
est = get_estimator(max_train_steps_per_epoch=300, epochs=4, log_steps=150, monitor_names="ce1")
est.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 150; 
    FastEstimator-Train: step: 1; ce1: 2.30421; ce: 2.30421; 
    FastEstimator-Train: step: 150; ce1: 0.35948867; ce: 0.35948867; steps/sec: 38.23; 
    FastEstimator-Train: step: 300; ce1: 0.16791707; ce: 0.16791707; steps/sec: 40.98; 
    FastEstimator-Train: step: 300; epoch: 1; epoch_time: 7.64 sec; 
    FastEstimator-Eval: step: 300; epoch: 1; ce1: 0.2302698; ce: 0.2302698; 
    FastEstimator-Train: step: 450; ce1: 0.14853987; ce: 0.14853987; steps/sec: 38.23; 
    FastEstimator-Train: step: 600; ce1: 0.49784163; ce: 0.49784163; steps/sec: 40.68; 
    FastEstimator-Train: step: 600; epoch: 2; epoch_time: 7.61 sec; 
    FastEstimator-Eval: step: 600; epoch: 2; ce1: 0.12643811; ce: 0.12643811; 
    FastEstimator-Train: step: 750; ce1: 0.18601; ce: 0.18601; steps/sec: 37.24; 
    FastEstimator-Train: step: 900; ce1: 0.12327108; ce: 0.12327108; steps/sec: 40.5; 
    FastEstimator-Train: step: 900; epoch: 3; epoch_time: 7.73 sec; 
    FastEstimator-Eval: step: 900; epoch: 3; ce1: 0.069144465; ce: 0.069144465; 
    FastEstimator-Train: step: 1050; ce1: 0.1580712; ce: 0.1580712; steps/sec: 37.91; 
    FastEstimator-Train: step: 1200; ce1: 0.20800333; ce: 0.20800333; steps/sec: 40.61; 
    FastEstimator-Train: step: 1200; epoch: 4; epoch_time: 7.65 sec; 
    FastEstimator-Eval: step: 1200; epoch: 4; ce1: 0.06323946; ce: 0.06323946; 
    FastEstimator-Finish: step: 1200; total_time: 37.49 sec; LeNet_lr: 0.001; 


As we can see, both `ce` and `ce1` showed up in the log above. Unsurprisingly, their values are identical because because they have the same inputs and forward function.

<a id='t07trace'></a>

## Trace

<a id='t07concept'></a>

### Concept
Now you might be thinking: 'changing logging behavior and monitoring extra keys is cool, but where is the fine-grained access to the training loop?' 

The answer is `Trace`. `Trace` is a module that can offer you access to different training stages and allow you "do stuff" with them. Here are some examples of what a `Trace` can do:

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

<img src="assets/branches/r1.0/tutorial/../resources/t07_trace_concept.png" alt="drawing" width="500"/>

As we can see from the illustration above, the training process is essentially a nested combination of batch loops and epoch loops. Over the course of training, `Trace` places 6 different "road blocks" for you to leverage.

<a id='t07structure'></a>

### Structure
If you are familiar with Keras, you will notice that the structure of `Trace` is very similar to the `Callback` in keras.  Despite the structural similarity, `Trace` gives you a lot more flexibility which we will talk about in depth in [advanced Tutorial 4](./tutorials/r1.0/advanced/t04_trace). Implementation-wise, `Trace` is a python class with the following structure:


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

Given the structure, users can customize their own functions at different stages and insert them into the training loop. We will leave the customization of `Traces` to the advanced tutorial. For now, let's use some pre-built `Traces` from FastEstimator.

During the training loop in our earlier example, we want 2 things to happen:
1. Save the model weights if the evaluation loss is the best we have seen so far
2. Calculate the model accuracy during evaluation

<a id='t07usage'></a>


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
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 100; 
    FastEstimator-Train: step: 1; ce: 2.317368; 
    FastEstimator-Train: step: 100; ce: 0.32270017; steps/sec: 38.37; 
    FastEstimator-Train: step: 200; ce: 0.4691573; steps/sec: 41.07; 
    FastEstimator-Train: step: 300; ce: 0.16797979; steps/sec: 41.48; 
    FastEstimator-Train: step: 400; ce: 0.22231343; steps/sec: 40.29; 
    FastEstimator-Train: step: 500; ce: 0.15864769; steps/sec: 40.23; 
    FastEstimator-Train: step: 600; ce: 0.21094382; steps/sec: 40.3; 
    FastEstimator-Train: step: 700; ce: 0.2174505; steps/sec: 39.09; 
    FastEstimator-Train: step: 800; ce: 0.1638605; steps/sec: 37.76; 
    FastEstimator-Train: step: 900; ce: 0.10876638; steps/sec: 38.04; 
    FastEstimator-Train: step: 1000; ce: 0.045762353; steps/sec: 37.84; 
    FastEstimator-Train: step: 1100; ce: 0.1986717; steps/sec: 37.9; 
    FastEstimator-Train: step: 1200; ce: 0.019097174; steps/sec: 38.52; 
    FastEstimator-Train: step: 1300; ce: 0.014496669; steps/sec: 38.07; 
    FastEstimator-Train: step: 1400; ce: 0.12824036; steps/sec: 37.98; 
    FastEstimator-Train: step: 1500; ce: 0.12543677; steps/sec: 37.89; 
    FastEstimator-Train: step: 1600; ce: 0.054099947; steps/sec: 38.18; 
    FastEstimator-Train: step: 1700; ce: 0.03653385; steps/sec: 38.03; 
    FastEstimator-Train: step: 1800; ce: 0.021161698; steps/sec: 38.84; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 48.39 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpe3yqgszs/LeNet_best_accuracy.pt
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.048291177; accuracy: 0.9846; since_best_accuracy: 0; max_accuracy: 0.9846; 
    FastEstimator-Train: step: 1900; ce: 0.05266206; steps/sec: 35.05; 
    FastEstimator-Train: step: 2000; ce: 0.010248414; steps/sec: 39.34; 
    FastEstimator-Train: step: 2100; ce: 0.100841954; steps/sec: 40.43; 
    FastEstimator-Train: step: 2200; ce: 0.099233195; steps/sec: 40.17; 
    FastEstimator-Train: step: 2300; ce: 0.014007135; steps/sec: 39.87; 
    FastEstimator-Train: step: 2400; ce: 0.100575976; steps/sec: 40.11; 
    FastEstimator-Train: step: 2500; ce: 0.014702196; steps/sec: 39.65; 
    FastEstimator-Train: step: 2600; ce: 0.017802792; steps/sec: 38.99; 
    FastEstimator-Train: step: 2700; ce: 0.07476275; steps/sec: 39.37; 
    FastEstimator-Train: step: 2800; ce: 0.0125279; steps/sec: 39.71; 
    FastEstimator-Train: step: 2900; ce: 0.02689986; steps/sec: 39.72; 
    FastEstimator-Train: step: 3000; ce: 0.00028639697; steps/sec: 38.95; 
    FastEstimator-Train: step: 3100; ce: 0.02897156; steps/sec: 37.94; 
    FastEstimator-Train: step: 3200; ce: 0.13989474; steps/sec: 38.29; 
    FastEstimator-Train: step: 3300; ce: 0.0010959036; steps/sec: 38.47; 
    FastEstimator-Train: step: 3400; ce: 0.014437494; steps/sec: 38.27; 
    FastEstimator-Train: step: 3500; ce: 0.13830313; steps/sec: 38.12; 
    FastEstimator-Train: step: 3600; ce: 0.0012470288; steps/sec: 38.18; 
    FastEstimator-Train: step: 3700; ce: 0.004030655; steps/sec: 38.39; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 48.25 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpe3yqgszs/LeNet_best_accuracy.pt
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.046231214; accuracy: 0.9854; since_best_accuracy: 0; max_accuracy: 0.9854; 
    FastEstimator-Finish: step: 3750; total_time: 100.12 sec; LeNet_lr: 0.001; 


As we can see from the log, the model is saved in a predefined location and the accuracy is displayed during evaluation.

<a id='t07testing'></a>

## Model Testing

Sometimes you have a separate testing dataset other than training and evaluation data. If you want to evalate the model metrics on test data, you can simply call: 


```python
est.test()
```

    FastEstimator-Test: step: 3750; epoch: 2; accuracy: 0.9844; 


This will feed all of your test dataset through the `Pipeline` and `Network`, and finally execute the traces (in our case, compute accuracy) just like during the training.

<a id='t07apphub'></a>

## Apphub Examples
You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [UNet](./examples/r1.0/semantic_segmentation/unet)
