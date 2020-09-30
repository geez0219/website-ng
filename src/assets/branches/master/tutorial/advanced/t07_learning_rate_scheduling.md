# Advanced Tutorial 7: Learning Rate Scheduling

## Overview

In this tutorial, we will discuss:

* [Customizing a Learning Rate Schedule Function](./tutorials/master/advanced/t07_learning_rate_scheduling#ta07customize)
    * [epoch-wise](./tutorials/master/advanced/t07_learning_rate_scheduling#ta07epoch)
    * [step-wise](./tutorials/master/advanced/t07_learning_rate_scheduling#ta07step)
* [Using a Built-In lr_schedule Function](./tutorials/master/advanced/t07_learning_rate_scheduling#ta07builtin)
    * [cosine decay](./tutorials/master/advanced/t07_learning_rate_scheduling#ta07cosine)
* [Related Apphub Examples](./tutorials/master/advanced/t07_learning_rate_scheduling#ta07apphub)

Learning rate schedules can be implemented using the `LRScheduler` `Trace`. `LRScheduler` takes the model and learning schedule through the **lr_fn** parameter. **lr_fn** should be a function/lambda function with 'step' or 'epoch' as its input parameter. This determines whether the learning schedule will be applied at a step or epoch level.

For more details on traces, you can visit [Beginner Tutorial 7](./tutorials/master/beginner/t07_estimator) and [Advanced Tutorial 4](./tutorials/master/advanced/t04_trace). 

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
To apply learning rate scheduling at an epoch level, the custom function should have 'epoch' as its parameter. Let's look at the example below which demonstrates this. We will be using the summary parameter in the fit method to be able to visualize the learning rate later. You can go through [Advanced Tutorial 6](./tutorials/master/advanced/t06_summary) for more details on accessing training history.


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
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 100; 
    FastEstimator-Train: step: 1; ce: 2.3089; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 100; ce: 0.58078986; steps/sec: 694.76; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 200; ce: 0.13996598; steps/sec: 767.68; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 300; ce: 0.047897074; steps/sec: 784.24; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 400; ce: 0.046643212; steps/sec: 776.27; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 500; ce: 0.022375159; steps/sec: 815.21; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 600; ce: 0.07842708; steps/sec: 778.69; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 700; ce: 0.20251414; steps/sec: 802.99; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 800; ce: 0.035366945; steps/sec: 769.72; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 900; ce: 0.03398672; steps/sec: 810.71; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1000; ce: 0.112584725; steps/sec: 783.55; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1100; ce: 0.05205777; steps/sec: 689.27; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1200; ce: 0.0033754208; steps/sec: 743.87; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1300; ce: 0.0054937536; steps/sec: 803.25; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1400; ce: 0.0065217884; steps/sec: 783.11; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1500; ce: 0.011019227; steps/sec: 819.06; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1600; ce: 0.05610779; steps/sec: 783.92; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1700; ce: 0.10374484; steps/sec: 812.64; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1800; ce: 0.16797249; steps/sec: 777.94; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 2.96 sec; 
    FastEstimator-Train: step: 1900; ce: 0.002968135; steps/sec: 456.57; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2000; ce: 0.004666821; steps/sec: 661.41; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2100; ce: 0.0124099245; steps/sec: 707.8; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2200; ce: 0.08333805; steps/sec: 765.97; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2300; ce: 0.04198639; steps/sec: 770.37; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2400; ce: 0.072333984; steps/sec: 788.7; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2500; ce: 0.0021644386; steps/sec: 783.02; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2600; ce: 0.117298014; steps/sec: 805.19; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2700; ce: 0.029399084; steps/sec: 787.32; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2800; ce: 0.025874225; steps/sec: 810.38; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2900; ce: 0.0076365666; steps/sec: 788.14; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3000; ce: 0.018179502; steps/sec: 793.57; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3100; ce: 0.002729386; steps/sec: 780.15; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3200; ce: 0.005655894; steps/sec: 785.53; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3300; ce: 0.0051174066; steps/sec: 772.76; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3400; ce: 0.03424426; steps/sec: 714.72; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3500; ce: 0.061904356; steps/sec: 692.78; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3600; ce: 0.01764475; steps/sec: 815.43; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3700; ce: 0.004598704; steps/sec: 796.09; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 2.54 sec; 
    FastEstimator-Train: step: 3800; ce: 0.007359849; steps/sec: 466.42; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 3900; ce: 0.03665335; steps/sec: 798.22; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4000; ce: 0.010769706; steps/sec: 775.86; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4100; ce: 0.0013347296; steps/sec: 794.47; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4200; ce: 0.00937571; steps/sec: 776.57; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4300; ce: 0.0073838052; steps/sec: 798.44; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4400; ce: 0.0016001706; steps/sec: 755.96; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4500; ce: 0.0027758705; steps/sec: 789.77; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4600; ce: 0.14900081; steps/sec: 771.62; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4700; ce: 0.00067295914; steps/sec: 794.2; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4800; ce: 0.035189193; steps/sec: 687.94; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4900; ce: 0.013106734; steps/sec: 725.63; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5000; ce: 0.0010486699; steps/sec: 790.38; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5100; ce: 0.0015635535; steps/sec: 801.9; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5200; ce: 0.22061187; steps/sec: 772.13; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5300; ce: 0.0050542983; steps/sec: 806.05; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5400; ce: 0.0024875803; steps/sec: 764.54; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5500; ce: 0.0076733916; steps/sec: 832.12; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5600; ce: 0.005431102; steps/sec: 804.78; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5625; epoch: 3; epoch_time: 2.54 sec; 
    FastEstimator-Finish: step: 5625; total_time: 9.77 sec; LeNet_lr: 0.0009; 


The learning rate is available in the training log at steps specified using the log_steps parameter in the `Estimator`. By default, training is logged every 100 steps.


```python
visualize_logs(history, include_metrics="LeNet_lr")
```


    
![png](assets/branches/master/tutorial/advanced/t07_learning_rate_scheduling_files/t07_learning_rate_scheduling_9_0.png)
    


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
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 100; 
    FastEstimator-Train: step: 1; ce: 2.3065164; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 100; ce: 0.23714758; steps/sec: 655.65; LeNet_lr: 0.0009868; 
    FastEstimator-Train: step: 200; ce: 0.6577442; steps/sec: 678.86; LeNet_lr: 0.00097346667; 
    FastEstimator-Train: step: 300; ce: 0.14811869; steps/sec: 677.92; LeNet_lr: 0.00096013333; 
    FastEstimator-Train: step: 400; ce: 0.11562818; steps/sec: 668.4; LeNet_lr: 0.0009468; 
    FastEstimator-Train: step: 500; ce: 0.027212799; steps/sec: 633.48; LeNet_lr: 0.00093346665; 
    FastEstimator-Train: step: 600; ce: 0.17180511; steps/sec: 570.52; LeNet_lr: 0.0009201333; 
    FastEstimator-Train: step: 700; ce: 0.060723193; steps/sec: 695.62; LeNet_lr: 0.0009068; 
    FastEstimator-Train: step: 800; ce: 0.072167784; steps/sec: 682.35; LeNet_lr: 0.00089346664; 
    FastEstimator-Train: step: 900; ce: 0.037193242; steps/sec: 683.86; LeNet_lr: 0.00088013336; 
    FastEstimator-Train: step: 1000; ce: 0.09921763; steps/sec: 605.18; LeNet_lr: 0.0008668; 
    FastEstimator-Train: step: 1100; ce: 0.050317485; steps/sec: 603.58; LeNet_lr: 0.0008534667; 
    FastEstimator-Train: step: 1200; ce: 0.033182904; steps/sec: 682.28; LeNet_lr: 0.00084013335; 
    FastEstimator-Train: step: 1300; ce: 0.030531863; steps/sec: 707.21; LeNet_lr: 0.0008268; 
    FastEstimator-Train: step: 1400; ce: 0.033350274; steps/sec: 683.06; LeNet_lr: 0.0008134667; 
    FastEstimator-Train: step: 1500; ce: 0.20844415; steps/sec: 706.99; LeNet_lr: 0.00080013333; 
    FastEstimator-Train: step: 1600; ce: 0.0029021623; steps/sec: 685.66; LeNet_lr: 0.0007868; 
    FastEstimator-Train: step: 1700; ce: 0.009277768; steps/sec: 717.94; LeNet_lr: 0.00077346666; 
    FastEstimator-Train: step: 1800; ce: 0.0021057532; steps/sec: 688.16; LeNet_lr: 0.0007601333; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 3.0 sec; 
    FastEstimator-Train: step: 1900; ce: 0.03702537; steps/sec: 417.6; LeNet_lr: 0.0007468; 
    FastEstimator-Train: step: 2000; ce: 0.053149987; steps/sec: 651.14; LeNet_lr: 0.00073346664; 
    FastEstimator-Train: step: 2100; ce: 0.017920867; steps/sec: 670.72; LeNet_lr: 0.0007201333; 
    FastEstimator-Train: step: 2200; ce: 0.08334316; steps/sec: 684.75; LeNet_lr: 0.0007068; 
    FastEstimator-Train: step: 2300; ce: 0.003692674; steps/sec: 682.05; LeNet_lr: 0.0006934667; 
    FastEstimator-Train: step: 2400; ce: 0.003553884; steps/sec: 690.95; LeNet_lr: 0.00068013335; 
    FastEstimator-Train: step: 2500; ce: 0.013678698; steps/sec: 661.92; LeNet_lr: 0.0006668; 
    FastEstimator-Train: step: 2600; ce: 0.07064867; steps/sec: 716.98; LeNet_lr: 0.0006534667; 
    FastEstimator-Train: step: 2700; ce: 0.036846854; steps/sec: 686.8; LeNet_lr: 0.00064013334; 
    FastEstimator-Train: step: 2800; ce: 0.004501665; steps/sec: 671.16; LeNet_lr: 0.0006268; 
    FastEstimator-Train: step: 2900; ce: 0.2406652; steps/sec: 566.73; LeNet_lr: 0.00061346666; 
    FastEstimator-Train: step: 3000; ce: 0.004612835; steps/sec: 703.64; LeNet_lr: 0.0006001333; 
    FastEstimator-Train: step: 3100; ce: 0.04271071; steps/sec: 568.53; LeNet_lr: 0.0005868; 
    FastEstimator-Train: step: 3200; ce: 0.124661796; steps/sec: 674.27; LeNet_lr: 0.00057346665; 
    FastEstimator-Train: step: 3300; ce: 0.012699548; steps/sec: 592.47; LeNet_lr: 0.0005601333; 
    FastEstimator-Train: step: 3400; ce: 0.0035635328; steps/sec: 636.74; LeNet_lr: 0.0005468; 
    FastEstimator-Train: step: 3500; ce: 0.116995685; steps/sec: 683.88; LeNet_lr: 0.0005334667; 
    FastEstimator-Train: step: 3600; ce: 0.007870817; steps/sec: 691.72; LeNet_lr: 0.00052013336; 
    FastEstimator-Train: step: 3700; ce: 0.01830994; steps/sec: 676.26; LeNet_lr: 0.0005068; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 2.96 sec; 
    FastEstimator-Finish: step: 3750; total_time: 7.12 sec; LeNet_lr: 0.0005001333; 



```python
visualize_logs(history2, include_metrics="LeNet_lr")
```


    
![png](assets/branches/master/tutorial/advanced/t07_learning_rate_scheduling_files/t07_learning_rate_scheduling_14_0.png)
    


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
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 100; 
    FastEstimator-Train: step: 1; ce: 2.2930279; LeNet_lr: 0.0009999993; 
    FastEstimator-Train: step: 100; ce: 0.26312476; steps/sec: 638.7; LeNet_lr: 0.000993005; 
    FastEstimator-Train: step: 200; ce: 0.12205046; steps/sec: 667.97; LeNet_lr: 0.000972216; 
    FastEstimator-Train: step: 300; ce: 0.14842911; steps/sec: 696.66; LeNet_lr: 0.0009382152; 
    FastEstimator-Train: step: 400; ce: 0.27469447; steps/sec: 674.11; LeNet_lr: 0.00089195487; 
    FastEstimator-Train: step: 500; ce: 0.01054606; steps/sec: 705.46; LeNet_lr: 0.00083473074; 
    FastEstimator-Train: step: 600; ce: 0.18344438; steps/sec: 686.91; LeNet_lr: 0.0007681455; 
    FastEstimator-Train: step: 700; ce: 0.029220346; steps/sec: 682.13; LeNet_lr: 0.000694064; 
    FastEstimator-Train: step: 800; ce: 0.13878524; steps/sec: 676.54; LeNet_lr: 0.00061456126; 
    FastEstimator-Train: step: 900; ce: 0.028735036; steps/sec: 698.14; LeNet_lr: 0.0005318639; 
    FastEstimator-Train: step: 1000; ce: 0.055955518; steps/sec: 667.36; LeNet_lr: 0.00044828805; 
    FastEstimator-Train: step: 1100; ce: 0.07805036; steps/sec: 699.72; LeNet_lr: 0.00036617456; 
    FastEstimator-Train: step: 1200; ce: 0.0090379715; steps/sec: 683.74; LeNet_lr: 0.00028782323; 
    FastEstimator-Train: step: 1300; ce: 0.077315584; steps/sec: 586.27; LeNet_lr: 0.00021542858; 
    FastEstimator-Train: step: 1400; ce: 0.05346773; steps/sec: 624.91; LeNet_lr: 0.00015101816; 
    FastEstimator-Train: step: 1500; ce: 0.08146605; steps/sec: 683.15; LeNet_lr: 9.639601e-05; 
    FastEstimator-Train: step: 1600; ce: 0.0033195266; steps/sec: 587.82; LeNet_lr: 5.3091975e-05; 
    FastEstimator-Train: step: 1700; ce: 0.12897912; steps/sec: 668.69; LeNet_lr: 2.231891e-05; 
    FastEstimator-Train: step: 1800; ce: 0.004909375; steps/sec: 673.23; LeNet_lr: 4.9387068e-06; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 3.07 sec; 
    FastEstimator-Train: step: 1900; ce: 0.15640017; steps/sec: 340.63; LeNet_lr: 0.0009995619; 
    FastEstimator-Train: step: 2000; ce: 0.089771196; steps/sec: 674.14; LeNet_lr: 0.0009890847; 
    FastEstimator-Train: step: 2100; ce: 0.013233781; steps/sec: 666.61; LeNet_lr: 0.00096492335; 
    FastEstimator-Train: step: 2200; ce: 0.026806809; steps/sec: 679.56; LeNet_lr: 0.00092775445; 
    FastEstimator-Train: step: 2300; ce: 0.011386171; steps/sec: 673.87; LeNet_lr: 0.00087861903; 
    FastEstimator-Train: step: 2400; ce: 0.018773185; steps/sec: 696.83; LeNet_lr: 0.00081889326; 
    FastEstimator-Train: step: 2500; ce: 0.03541358; steps/sec: 671.42; LeNet_lr: 0.00075025; 
    FastEstimator-Train: step: 2600; ce: 0.11294758; steps/sec: 714.27; LeNet_lr: 0.0006746117; 
    FastEstimator-Train: step: 2700; ce: 0.028423183; steps/sec: 697.1; LeNet_lr: 0.00059409696; 
    FastEstimator-Train: step: 2800; ce: 0.16977276; steps/sec: 703.42; LeNet_lr: 0.00051096076; 
    FastEstimator-Train: step: 2900; ce: 0.01733088; steps/sec: 669.76; LeNet_lr: 0.00042753152; 
    FastEstimator-Train: step: 3000; ce: 0.02646891; steps/sec: 680.62; LeNet_lr: 0.000346146; 
    FastEstimator-Train: step: 3100; ce: 0.002477122; steps/sec: 594.12; LeNet_lr: 0.00026908363; 
    FastEstimator-Train: step: 3200; ce: 0.013257658; steps/sec: 621.69; LeNet_lr: 0.00019850275; 
    FastEstimator-Train: step: 3300; ce: 0.003857479; steps/sec: 684.61; LeNet_lr: 0.00013638017; 
    FastEstimator-Train: step: 3400; ce: 0.029402707; steps/sec: 692.26; LeNet_lr: 8.445584e-05; 
    FastEstimator-Train: step: 3500; ce: 0.00379532; steps/sec: 671.9; LeNet_lr: 4.4184046e-05; 
    FastEstimator-Train: step: 3600; ce: 0.0058736084; steps/sec: 705.67; LeNet_lr: 1.6692711e-05; 
    FastEstimator-Train: step: 3700; ce: 0.031074949; steps/sec: 670.42; LeNet_lr: 2.7518167e-06; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 2.87 sec; 
    FastEstimator-Finish: step: 3750; total_time: 7.14 sec; LeNet_lr: 1e-06; 



```python
visualize_logs(history3, include_metrics="LeNet_lr")
```


    
![png](assets/branches/master/tutorial/advanced/t07_learning_rate_scheduling_files/t07_learning_rate_scheduling_20_0.png)
    


<a id='ta07apphub'></a>

## Apphub Examples
You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [MNIST](./examples/master/image_classification/mnist)
* [CIFAR10](./examples/master/image_classification/cifar10_fast)
