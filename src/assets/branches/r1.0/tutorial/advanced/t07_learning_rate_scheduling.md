# Advanced Tutorial 7: Learning Rate Scheduling

## Overview

In this tutorial, we will discuss:

* [Customizing a Learning Rate Schedule Function](#ta07customize)
    * [epoch-wise](#ta07epoch)
    * [step-wise](#ta07step)
* [Using a Built-In lr_schedule Function](#ta07builtin)
    * [cosine decay](#ta07cosine)
* [Related Apphub Examples](#ta07apphub)

Learning rate schedules can be implemented using the `LRScheduler` `Trace`. `LRScheduler` takes the model and learning schedule through the **lr_fn** parameter. **lr_fn** should be a function/lambda function with 'step' or 'epoch' as its input parameter. This determines whether the learning schedule will be applied at a step or epoch level.

For more details on traces, you can visit [tutorial 7](./tutorials/beginner/t07_estimator) in the beginner section and [tutorial 4](./tutorials/advanced/t04_trace) in the advanced section. 

Let's create a function to generate the pipeline, model, and network to be used for this tutorial:


```python
import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp


def get_pipeline_model_network(model_name="LeNet"):
    train_data, _ = mnist.load_data()

    pipeline = fe.Pipeline(train_data=train_data,
                           batch_size=32,
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

<a id='ta07customize'></a>

## Customizing a Learning Rate Schedule Function
We can specify a custom learning schedule by passing a custom function to the **lr_fn** parameter of `LRScheduler`. We can have this learning rate schedule applied at either the epoch or step level. Epoch and step both start from 1.

<a id='ta07epoch'></a>

### Epoch-wise
To apply learning rate scheduling at an epoch level, the custom function should have 'epoch' as its parameter. Let's look at the example below which demonstrates this. We will be using the summary parameter in the fit method to be able to visualize the learning rate later. You can go through [tutorial 6](./tutorials/advanced/t06_summary) in the advanced section for more details on accessing training history.


```python
from fastestimator.summary.logs import visualize_logs
from fastestimator.trace.adapt import LRScheduler

def lr_schedule(epoch):
    lr = 0.001*(20-epoch+1)/20
    return lr

pipeline, model, network = get_pipeline_model_network()

traces = LRScheduler(model=model, lr_fn=lr_schedule)
estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=3, traces=traces)

history = estimator.fit(summary="Experiment_1")
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 100; 
    FastEstimator-Train: step: 1; ce: 2.3121834; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 100; ce: 0.3843257; steps/sec: 139.02; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 200; ce: 0.117751315; steps/sec: 131.96; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 300; ce: 0.20433763; steps/sec: 129.58; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 400; ce: 0.10046323; steps/sec: 122.35; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 500; ce: 0.18251789; steps/sec: 122.15; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 600; ce: 0.027669977; steps/sec: 118.68; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 700; ce: 0.018796597; steps/sec: 117.34; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 800; ce: 0.1343742; steps/sec: 113.8; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 900; ce: 0.066348195; steps/sec: 101.95; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1000; ce: 0.21500783; steps/sec: 103.25; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1100; ce: 0.020392025; steps/sec: 101.94; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1200; ce: 0.06266867; steps/sec: 101.34; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1300; ce: 0.0051358286; steps/sec: 98.58; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1400; ce: 0.11623003; steps/sec: 95.29; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1500; ce: 0.2841274; steps/sec: 94.73; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1600; ce: 0.0059423; steps/sec: 91.64; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1700; ce: 0.020643737; steps/sec: 93.76; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1800; ce: 0.05520491; steps/sec: 94.92; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 19.52 sec; 
    FastEstimator-Train: step: 1900; ce: 0.011271543; steps/sec: 94.66; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2000; ce: 0.02188245; steps/sec: 89.22; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2100; ce: 0.0048960065; steps/sec: 84.91; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2200; ce: 0.009766675; steps/sec: 90.6; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2300; ce: 0.008525206; steps/sec: 88.56; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2400; ce: 0.007846909; steps/sec: 85.83; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2500; ce: 0.11211253; steps/sec: 84.11; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2600; ce: 0.013980574; steps/sec: 80.19; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2700; ce: 0.026221849; steps/sec: 79.1; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2800; ce: 0.025860263; steps/sec: 78.37; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2900; ce: 0.008109703; steps/sec: 77.45; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3000; ce: 0.09107354; steps/sec: 75.67; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3100; ce: 0.003782929; steps/sec: 73.88; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3200; ce: 0.0090578655; steps/sec: 72.33; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3300; ce: 0.103154205; steps/sec: 83.93; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3400; ce: 0.013315021; steps/sec: 79.49; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3500; ce: 0.017441805; steps/sec: 85.13; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3600; ce: 0.017847143; steps/sec: 77.92; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3700; ce: 0.04056422; steps/sec: 82.93; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 23.04 sec; 
    FastEstimator-Train: step: 3800; ce: 0.004228523; steps/sec: 79.59; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 3900; ce: 0.0015197117; steps/sec: 81.38; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4000; ce: 0.0004167799; steps/sec: 79.58; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4100; ce: 0.012444577; steps/sec: 79.87; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4200; ce: 0.011467964; steps/sec: 80.24; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4300; ce: 0.004452401; steps/sec: 76.3; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4400; ce: 0.0045919186; steps/sec: 77.09; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4500; ce: 0.05509679; steps/sec: 80.72; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4600; ce: 0.0028458317; steps/sec: 76.64; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4700; ce: 0.014053181; steps/sec: 78.09; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4800; ce: 0.00821719; steps/sec: 77.62; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4900; ce: 0.05650976; steps/sec: 81.53; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5000; ce: 0.044777524; steps/sec: 75.98; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5100; ce: 0.000998376; steps/sec: 76.29; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5200; ce: 0.00041511073; steps/sec: 73.71; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5300; ce: 0.023241056; steps/sec: 86.19; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5400; ce: 0.18524896; steps/sec: 81.82; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5500; ce: 0.12962568; steps/sec: 83.83; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5600; ce: 0.0027833423; steps/sec: 83.36; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5625; epoch: 3; epoch_time: 23.63 sec; 
    FastEstimator-Finish: step: 5625; total_time: 66.27 sec; LeNet_lr: 0.0009; 


The learning rate is available in the training log at steps specified using the log_steps parameter in the `Estimator`. By default, training is logged every 100 steps.


```python
visualize_logs(history, include_metrics="LeNet_lr")
```


![png](assets/branches/r1.0/tutorial/advanced/t07_learning_rate_scheduling_files/t07_learning_rate_scheduling_9_0.png)


As you can see, the learning rate changes only after every epoch.

<a id='ta07step'></a>

### Step-wise
The custom function should have 'step' as its parameter for step-based learning rate schedules. 


```python
def lr_schedule(step):
    lr = 0.001*(7500-step+1)/7500
    return lr

pipeline, model, network = get_pipeline_model_network()

traces = LRScheduler(model=model, lr_fn=lr_schedule)
estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=2, traces=traces)

history2 = estimator.fit(summary="Experiment_2")
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 100; 
    FastEstimator-Train: step: 1; ce: 2.3268642; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 100; ce: 0.34729302; steps/sec: 81.02; LeNet_lr: 0.0009868; 
    FastEstimator-Train: step: 200; ce: 0.16947752; steps/sec: 79.79; LeNet_lr: 0.00097346667; 
    FastEstimator-Train: step: 300; ce: 0.1158019; steps/sec: 79.11; LeNet_lr: 0.00096013333; 
    FastEstimator-Train: step: 400; ce: 0.08662731; steps/sec: 77.06; LeNet_lr: 0.0009468; 
    FastEstimator-Train: step: 500; ce: 0.0455483; steps/sec: 75.55; LeNet_lr: 0.00093346665; 
    FastEstimator-Train: step: 600; ce: 0.17311177; steps/sec: 77.16; LeNet_lr: 0.0009201333; 
    FastEstimator-Train: step: 700; ce: 0.07044801; steps/sec: 77.25; LeNet_lr: 0.0009068; 
    FastEstimator-Train: step: 800; ce: 0.060867704; steps/sec: 76.46; LeNet_lr: 0.00089346664; 
    FastEstimator-Train: step: 900; ce: 0.14833035; steps/sec: 77.65; LeNet_lr: 0.00088013336; 
    FastEstimator-Train: step: 1000; ce: 0.01502298; steps/sec: 75.22; LeNet_lr: 0.0008668; 
    FastEstimator-Train: step: 1100; ce: 0.14445232; steps/sec: 75.63; LeNet_lr: 0.0008534667; 
    FastEstimator-Train: step: 1200; ce: 0.027258653; steps/sec: 74.33; LeNet_lr: 0.00084013335; 
    FastEstimator-Train: step: 1300; ce: 0.12475329; steps/sec: 73.37; LeNet_lr: 0.0008268; 
    FastEstimator-Train: step: 1400; ce: 0.09298558; steps/sec: 76.89; LeNet_lr: 0.0008134667; 
    FastEstimator-Train: step: 1500; ce: 0.048965212; steps/sec: 71.28; LeNet_lr: 0.00080013333; 
    FastEstimator-Train: step: 1600; ce: 0.043585315; steps/sec: 74.56; LeNet_lr: 0.0007868; 
    FastEstimator-Train: step: 1700; ce: 0.18490058; steps/sec: 72.17; LeNet_lr: 0.00077346666; 
    FastEstimator-Train: step: 1800; ce: 0.08914224; steps/sec: 73.68; LeNet_lr: 0.0007601333; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 24.87 sec; 
    FastEstimator-Train: step: 1900; ce: 0.06549401; steps/sec: 72.05; LeNet_lr: 0.0007468; 
    FastEstimator-Train: step: 2000; ce: 0.005398229; steps/sec: 73.46; LeNet_lr: 0.00073346664; 
    FastEstimator-Train: step: 2100; ce: 0.19036639; steps/sec: 73.44; LeNet_lr: 0.0007201333; 
    FastEstimator-Train: step: 2200; ce: 0.0037528607; steps/sec: 74.23; LeNet_lr: 0.0007068; 
    FastEstimator-Train: step: 2300; ce: 0.07193564; steps/sec: 73.72; LeNet_lr: 0.0006934667; 
    FastEstimator-Train: step: 2400; ce: 0.27834976; steps/sec: 72.84; LeNet_lr: 0.00068013335; 
    FastEstimator-Train: step: 2500; ce: 0.058940012; steps/sec: 74.08; LeNet_lr: 0.0006668; 
    FastEstimator-Train: step: 2600; ce: 0.14127687; steps/sec: 73.21; LeNet_lr: 0.0006534667; 
    FastEstimator-Train: step: 2700; ce: 0.03426341; steps/sec: 75.49; LeNet_lr: 0.00064013334; 
    FastEstimator-Train: step: 2800; ce: 0.033499897; steps/sec: 71.58; LeNet_lr: 0.0006268; 
    FastEstimator-Train: step: 2900; ce: 0.008997633; steps/sec: 75.63; LeNet_lr: 0.00061346666; 
    FastEstimator-Train: step: 3000; ce: 0.02539826; steps/sec: 74.05; LeNet_lr: 0.0006001333; 
    FastEstimator-Train: step: 3100; ce: 0.10326672; steps/sec: 72.52; LeNet_lr: 0.0005868; 
    FastEstimator-Train: step: 3200; ce: 0.008950228; steps/sec: 78.7; LeNet_lr: 0.00057346665; 
    FastEstimator-Train: step: 3300; ce: 0.023044724; steps/sec: 76.58; LeNet_lr: 0.0005601333; 
    FastEstimator-Train: step: 3400; ce: 0.030512333; steps/sec: 72.12; LeNet_lr: 0.0005468; 
    FastEstimator-Train: step: 3500; ce: 0.0038094707; steps/sec: 72.78; LeNet_lr: 0.0005334667; 
    FastEstimator-Train: step: 3600; ce: 0.041347966; steps/sec: 75.93; LeNet_lr: 0.00052013336; 
    FastEstimator-Train: step: 3700; ce: 0.0005119173; steps/sec: 72.98; LeNet_lr: 0.0005068; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 25.35 sec; 
    FastEstimator-Finish: step: 3750; total_time: 50.26 sec; LeNet_lr: 0.0005001333; 



```python
visualize_logs(history2, include_metrics="LeNet_lr")
```


![png](assets/branches/r1.0/tutorial/advanced/t07_learning_rate_scheduling_files/t07_learning_rate_scheduling_14_0.png)


<a id='ta07builtin'></a>

## Using Built-In lr_schedule Function
Some learning rates schedules are widely popular in the deep learning community. We have implemented some of them in FastEstimator so that you don't need to write a custom schedule for them. We will be showcasing the `cosine decay` schedule below.

<a id='ta07cosine'></a>

### cosine_decay
We can specify the length of the decay cycle and initial learning rate using cycle_length and init_lr respectively. Similar to custom learning schedule, lr_fn should have step or epoch as a parameter. The FastEstimator cosine decay can be used as follows:


```python
from fastestimator.schedule import cosine_decay

pipeline, model, network = get_pipeline_model_network()

traces = LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=1875, init_lr=1e-3))
estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=2, traces=traces)

history3 = estimator.fit(summary="Experiment_3")
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 100; 
    FastEstimator-Train: step: 1; ce: 2.3090043; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 100; ce: 0.34759182; steps/sec: 72.64; LeNet_lr: 0.0009931439; 
    FastEstimator-Train: step: 200; ce: 0.124293044; steps/sec: 72.99; LeNet_lr: 0.0009724906; 
    FastEstimator-Train: step: 300; ce: 0.11162343; steps/sec: 76.42; LeNet_lr: 0.00093861774; 
    FastEstimator-Train: step: 400; ce: 0.1846501; steps/sec: 73.36; LeNet_lr: 0.0008924742; 
    FastEstimator-Train: step: 500; ce: 0.09742119; steps/sec: 72.06; LeNet_lr: 0.0008353522; 
    FastEstimator-Train: step: 600; ce: 0.013896314; steps/sec: 74.61; LeNet_lr: 0.0007688517; 
    FastEstimator-Train: step: 700; ce: 0.10306329; steps/sec: 71.42; LeNet_lr: 0.00069483527; 
    FastEstimator-Train: step: 800; ce: 0.020540427; steps/sec: 72.34; LeNet_lr: 0.00061537593; 
    FastEstimator-Train: step: 900; ce: 0.33192694; steps/sec: 74.66; LeNet_lr: 0.0005326991; 
    FastEstimator-Train: step: 1000; ce: 0.027314447; steps/sec: 73.71; LeNet_lr: 0.00044912045; 
    FastEstimator-Train: step: 1100; ce: 0.07889113; steps/sec: 71.89; LeNet_lr: 0.00036698082; 
    FastEstimator-Train: step: 1200; ce: 0.13510469; steps/sec: 71.83; LeNet_lr: 0.0002885808; 
    FastEstimator-Train: step: 1300; ce: 0.026300007; steps/sec: 75.5; LeNet_lr: 0.00021611621; 
    FastEstimator-Train: step: 1400; ce: 0.028623505; steps/sec: 70.93; LeNet_lr: 0.00015161661; 
    FastEstimator-Train: step: 1500; ce: 0.12303919; steps/sec: 72.35; LeNet_lr: 9.688851e-05; 
    FastEstimator-Train: step: 1600; ce: 0.08176289; steps/sec: 73.9; LeNet_lr: 5.3464726e-05; 
    FastEstimator-Train: step: 1700; ce: 0.018114068; steps/sec: 76.93; LeNet_lr: 2.2561479e-05; 
    FastEstimator-Train: step: 1800; ce: 0.02731928; steps/sec: 73.86; LeNet_lr: 5.0442964e-06; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 25.59 sec; 
    FastEstimator-Train: step: 1900; ce: 0.014814988; steps/sec: 76.8; LeNet_lr: 0.0009995962; 
    FastEstimator-Train: step: 2000; ce: 0.054916177; steps/sec: 76.62; LeNet_lr: 0.000989258; 
    FastEstimator-Train: step: 2100; ce: 0.016524037; steps/sec: 75.3; LeNet_lr: 0.0009652308; 
    FastEstimator-Train: step: 2200; ce: 0.03584575; steps/sec: 77.71; LeNet_lr: 0.0009281874; 
    FastEstimator-Train: step: 2300; ce: 0.016135138; steps/sec: 73.3; LeNet_lr: 0.00087916537; 
    FastEstimator-Train: step: 2400; ce: 0.017202383; steps/sec: 76.79; LeNet_lr: 0.0008195377; 
    FastEstimator-Train: step: 2500; ce: 0.02264627; steps/sec: 75.4; LeNet_lr: 0.00075097446; 
    FastEstimator-Train: step: 2600; ce: 0.014660467; steps/sec: 81.77; LeNet_lr: 0.00067539595; 
    FastEstimator-Train: step: 2700; ce: 0.1418988; steps/sec: 76.47; LeNet_lr: 0.0005949189; 
    FastEstimator-Train: step: 2800; ce: 0.033834495; steps/sec: 74.0; LeNet_lr: 0.00051179744; 
    FastEstimator-Train: step: 2900; ce: 0.117458574; steps/sec: 75.8; LeNet_lr: 0.00042835958; 
    FastEstimator-Train: step: 3000; ce: 0.039562702; steps/sec: 74.06; LeNet_lr: 0.0003469422; 
    FastEstimator-Train: step: 3100; ce: 0.013925586; steps/sec: 77.25; LeNet_lr: 0.00026982563; 
    FastEstimator-Train: step: 3200; ce: 0.010165918; steps/sec: 74.82; LeNet_lr: 0.0001991698; 
    FastEstimator-Train: step: 3300; ce: 0.009503109; steps/sec: 74.25; LeNet_lr: 0.00013695359; 
    FastEstimator-Train: step: 3400; ce: 0.02341089; steps/sec: 75.41; LeNet_lr: 8.491957e-05; 
    FastEstimator-Train: step: 3500; ce: 0.06209151; steps/sec: 74.37; LeNet_lr: 4.452509e-05; 
    FastEstimator-Train: step: 3600; ce: 0.105473876; steps/sec: 78.39; LeNet_lr: 1.6901524e-05; 
    FastEstimator-Train: step: 3700; ce: 0.010678038; steps/sec: 74.96; LeNet_lr: 2.8225472e-06; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 24.75 sec; 
    FastEstimator-Finish: step: 3750; total_time: 50.39 sec; LeNet_lr: 1.0007011e-06; 



```python
visualize_logs(history3, include_metrics="LeNet_lr")
```


![png](assets/branches/r1.0/tutorial/advanced/t07_learning_rate_scheduling_files/t07_learning_rate_scheduling_20_0.png)


<a id='ta07apphub'></a>

## Apphub Examples
You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [MNIST](./examples/image_classification/mnist)
* [CIFAR10](./examples/image_classification/cifar10_fast)
