# Advanced Tutorial 7: Learning Rate Scheduling

## Overview
In this tutorial we will talk about:
* **Customizing Learning Rate schedule function**
    * epoch-wise
    * step-wise
* **Using built-in lr_schedule function**
    * cosine decay

Learning rate schedules can be implemented using **LRScheduler** trace. LRScheduler takes the model and learning schedule through the **lr_fn** parameter. **lr_fn** should be a function/lambda function with step or epoch as parameter. This determines whether learning schedule will be applied at a step or epoch level.

For more details on traces, you can visit [tutorial 7](https://github.com/fastestimator/fastestimator/blob/master/tutorial/beginner/t07_estimator.ipynb) in beginner section and [tutorial 4](https://github.com/fastestimator/fastestimator/blob/master/tutorial/advanced/t04_trace.ipynb) in the advanced section. 

Let's create a function to generate pipeline, model and network to be used for the tutorial


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

    model = fe.build(model_fn=LeNet, optimizer_fn="adam", model_names=model_name)

    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    return pipeline, model, network
```

## Customizing Learning Rate schedule function
We can specify custom learning schedule by passing a custom function to the **lr_fn** parameter of LRScheduler. We can specify learning rate schedule to be applied at epoch or step level. Epoch and step start from 1.

### Epoch-wise
To apply learning rate at epoch level, the custom function should have epoch as a parameter. Let's look at the example below which demonstrates this. We will be using summary parameter in the fit method to be able to visualize the learning rate later. You can go through [Tutorial 6](https://github.com/fastestimator/fastestimator/blob/master/tutorial/advanced/t06_summary.ipynb) in the advanced section for more details on accessing training history.


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

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.2990923; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 100; ce: 0.4943217; steps/sec: 248.6; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 200; ce: 0.21011522; steps/sec: 251.18; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 300; ce: 0.050100736; steps/sec: 251.31; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 400; ce: 0.022691462; steps/sec: 254.55; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 500; ce: 0.09411738; steps/sec: 247.47; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 600; ce: 0.029331207; steps/sec: 257.27; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 700; ce: 0.14727665; steps/sec: 247.75; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 800; ce: 0.17017518; steps/sec: 246.74; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 900; ce: 0.06674249; steps/sec: 251.61; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1000; ce: 0.0123287; steps/sec: 244.46; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1100; ce: 0.042844594; steps/sec: 259.93; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1200; ce: 0.0058470815; steps/sec: 262.87; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1300; ce: 0.0049162265; steps/sec: 257.5; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1400; ce: 0.06464251; steps/sec: 247.75; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1500; ce: 0.061915822; steps/sec: 255.2; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1600; ce: 0.088724926; steps/sec: 244.9; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1700; ce: 0.040655892; steps/sec: 249.36; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1800; ce: 0.022507608; steps/sec: 253.53; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 10.84 sec; 
    FastEstimator-Train: step: 1900; ce: 0.0052600736; steps/sec: 174.9; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2000; ce: 0.16969691; steps/sec: 242.81; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2100; ce: 0.020830456; steps/sec: 246.42; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2200; ce: 0.17682028; steps/sec: 256.27; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2300; ce: 0.1840077; steps/sec: 259.81; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2400; ce: 0.0011837531; steps/sec: 255.83; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2500; ce: 0.0047405153; steps/sec: 254.61; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2600; ce: 0.25417805; steps/sec: 253.63; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2700; ce: 0.0009929237; steps/sec: 256.21; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2800; ce: 0.03668064; steps/sec: 261.31; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 2900; ce: 0.013435603; steps/sec: 261.73; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3000; ce: 0.031477045; steps/sec: 260.48; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3100; ce: 0.07136849; steps/sec: 244.6; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3200; ce: 0.050063; steps/sec: 250.05; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3300; ce: 0.020793993; steps/sec: 257.2; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3400; ce: 0.002888149; steps/sec: 248.12; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3500; ce: 0.005558454; steps/sec: 269.1; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3600; ce: 0.20430116; steps/sec: 256.44; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3700; ce: 0.047323182; steps/sec: 253.55; LeNet_lr: 0.00095; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 7.51 sec; 
    FastEstimator-Train: step: 3800; ce: 0.002140337; steps/sec: 184.72; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 3900; ce: 0.0059989765; steps/sec: 255.69; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4000; ce: 0.0024572595; steps/sec: 267.32; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4100; ce: 0.00391402; steps/sec: 255.03; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4200; ce: 0.0059940103; steps/sec: 271.17; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4300; ce: 0.016653711; steps/sec: 280.76; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4400; ce: 0.017520636; steps/sec: 281.83; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4500; ce: 0.013174945; steps/sec: 251.44; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4600; ce: 0.037972223; steps/sec: 272.66; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4700; ce: 0.0007721542; steps/sec: 263.75; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4800; ce: 0.046452045; steps/sec: 270.72; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 4900; ce: 0.0039830944; steps/sec: 270.78; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5000; ce: 0.008916434; steps/sec: 266.78; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5100; ce: 0.03087433; steps/sec: 243.26; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5200; ce: 0.007752375; steps/sec: 259.18; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5300; ce: 0.002660771; steps/sec: 261.86; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5400; ce: 0.04669408; steps/sec: 247.17; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5500; ce: 0.015778821; steps/sec: 261.74; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5600; ce: 0.00071525737; steps/sec: 267.46; LeNet_lr: 0.0009; 
    FastEstimator-Train: step: 5625; epoch: 3; epoch_time: 7.28 sec; 
    FastEstimator-Finish: step: 5625; total_time: 25.7 sec; LeNet_lr: 0.0009; 


The learning rate is available during training log at steps specified using log_steps parameter in the estimator. By default, training is logged at every 100 steps.


```python
visualize_logs(history, include_metrics="LeNet_lr")
```


![png](assets/tutorial/t07_learning_rate_scheduling_files/t07_learning_rate_scheduling_7_0.png)


As you can see, the learning rate changes only at an epoch level

### Step-wise
The custom function should have step as a parameter for step based learning schedules. 


```python
def lr_schedule(step):
    lr = 0.001*(7500-step+1)/7500
    return lr

pipeline, model, network = get_pipeline_model_network()

traces = LRScheduler(model=model, lr_fn=lr_schedule)
estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=2, traces=traces)

history2 = estimator.fit(summary="Experiment_2")
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.2912364; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 100; ce: 0.31862932; steps/sec: 240.19; LeNet_lr: 0.0009868; 
    FastEstimator-Train: step: 200; ce: 0.34517264; steps/sec: 239.47; LeNet_lr: 0.00097346667; 
    FastEstimator-Train: step: 300; ce: 0.12563829; steps/sec: 237.09; LeNet_lr: 0.00096013333; 
    FastEstimator-Train: step: 400; ce: 0.19191661; steps/sec: 240.71; LeNet_lr: 0.0009468; 
    FastEstimator-Train: step: 500; ce: 0.068875596; steps/sec: 231.16; LeNet_lr: 0.00093346665; 
    FastEstimator-Train: step: 600; ce: 0.023885677; steps/sec: 238.5; LeNet_lr: 0.0009201333; 
    FastEstimator-Train: step: 700; ce: 0.018282803; steps/sec: 241.55; LeNet_lr: 0.0009068; 
    FastEstimator-Train: step: 800; ce: 0.08792193; steps/sec: 229.97; LeNet_lr: 0.00089346664; 
    FastEstimator-Train: step: 900; ce: 0.05016824; steps/sec: 233.92; LeNet_lr: 0.00088013336; 
    FastEstimator-Train: step: 1000; ce: 0.034121446; steps/sec: 232.64; LeNet_lr: 0.0008668; 
    FastEstimator-Train: step: 1100; ce: 0.16328412; steps/sec: 222.7; LeNet_lr: 0.0008534667; 
    FastEstimator-Train: step: 1200; ce: 0.007584372; steps/sec: 237.0; LeNet_lr: 0.00084013335; 
    FastEstimator-Train: step: 1300; ce: 0.03577161; steps/sec: 242.67; LeNet_lr: 0.0008268; 
    FastEstimator-Train: step: 1400; ce: 0.07177823; steps/sec: 253.82; LeNet_lr: 0.0008134667; 
    FastEstimator-Train: step: 1500; ce: 0.015050432; steps/sec: 234.45; LeNet_lr: 0.00080013333; 
    FastEstimator-Train: step: 1600; ce: 0.01172007; steps/sec: 244.1; LeNet_lr: 0.0007868; 
    FastEstimator-Train: step: 1700; ce: 0.016387302; steps/sec: 232.82; LeNet_lr: 0.00077346666; 
    FastEstimator-Train: step: 1800; ce: 0.009856557; steps/sec: 238.4; LeNet_lr: 0.0007601333; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 8.25 sec; 
    FastEstimator-Train: step: 1900; ce: 0.0064330804; steps/sec: 174.58; LeNet_lr: 0.0007468; 
    FastEstimator-Train: step: 2000; ce: 0.048888315; steps/sec: 237.94; LeNet_lr: 0.00073346664; 
    FastEstimator-Train: step: 2100; ce: 0.08030538; steps/sec: 241.24; LeNet_lr: 0.0007201333; 
    FastEstimator-Train: step: 2200; ce: 0.061011378; steps/sec: 248.17; LeNet_lr: 0.0007068; 
    FastEstimator-Train: step: 2300; ce: 0.0112165585; steps/sec: 232.71; LeNet_lr: 0.0006934667; 
    FastEstimator-Train: step: 2400; ce: 0.006057879; steps/sec: 240.84; LeNet_lr: 0.00068013335; 
    FastEstimator-Train: step: 2500; ce: 0.06513041; steps/sec: 234.85; LeNet_lr: 0.0006668; 
    FastEstimator-Train: step: 2600; ce: 0.0049754623; steps/sec: 247.73; LeNet_lr: 0.0006534667; 
    FastEstimator-Train: step: 2700; ce: 0.011101577; steps/sec: 244.5; LeNet_lr: 0.00064013334; 
    FastEstimator-Train: step: 2800; ce: 0.027999522; steps/sec: 249.39; LeNet_lr: 0.0006268; 
    FastEstimator-Train: step: 2900; ce: 0.014848273; steps/sec: 253.34; LeNet_lr: 0.00061346666; 
    FastEstimator-Train: step: 3000; ce: 0.030323116; steps/sec: 246.26; LeNet_lr: 0.0006001333; 
    FastEstimator-Train: step: 3100; ce: 0.062911615; steps/sec: 252.46; LeNet_lr: 0.0005868; 
    FastEstimator-Train: step: 3200; ce: 0.037219964; steps/sec: 270.18; LeNet_lr: 0.00057346665; 
    FastEstimator-Train: step: 3300; ce: 0.04192976; steps/sec: 241.01; LeNet_lr: 0.0005601333; 
    FastEstimator-Train: step: 3400; ce: 0.16633682; steps/sec: 236.76; LeNet_lr: 0.0005468; 
    FastEstimator-Train: step: 3500; ce: 0.013411999; steps/sec: 240.79; LeNet_lr: 0.0005334667; 
    FastEstimator-Train: step: 3600; ce: 0.14039548; steps/sec: 248.67; LeNet_lr: 0.00052013336; 
    FastEstimator-Train: step: 3700; ce: 0.0046143783; steps/sec: 246.65; LeNet_lr: 0.0005068; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 7.82 sec; 
    FastEstimator-Finish: step: 3750; total_time: 16.11 sec; LeNet_lr: 0.0005001333; 



```python
visualize_logs(history2, include_metrics="LeNet_lr")
```


![png](assets/tutorial/t07_learning_rate_scheduling_files/t07_learning_rate_scheduling_11_0.png)


## Using built-in lr_schedule function
Some learning rates schedules are widely popular in the deep learning community. So, we have implemented some of them in fastestimator so that you don't need to write a custom schedule for them. We will be showcasing `cosine decay` schedule below.

### cosine_decay
We can specify the length of the decay cycle and initial learning rate using cycle_length and init_lr respectively. Similar to custom learning schedule, lr_fn should have step or epoch as a parameter. Implementation of cosine decay is shown below:


```python
from fastestimator.schedule import cosine_decay

pipeline, model, network = get_pipeline_model_network()

traces = LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=1875, init_lr=1e-3))
estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=2, traces=traces)

history3 = estimator.fit(summary="Experiment_3")
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.30225; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 100; ce: 0.5218753; steps/sec: 230.78; LeNet_lr: 0.0009931439; 
    FastEstimator-Train: step: 200; ce: 0.035484873; steps/sec: 244.72; LeNet_lr: 0.0009724906; 
    FastEstimator-Train: step: 300; ce: 0.17100406; steps/sec: 236.97; LeNet_lr: 0.00093861774; 
    FastEstimator-Train: step: 400; ce: 0.07232281; steps/sec: 238.56; LeNet_lr: 0.0008924742; 
    FastEstimator-Train: step: 500; ce: 0.36837792; steps/sec: 233.52; LeNet_lr: 0.0008353522; 
    FastEstimator-Train: step: 600; ce: 0.013588531; steps/sec: 237.59; LeNet_lr: 0.0007688517; 
    FastEstimator-Train: step: 700; ce: 0.073512316; steps/sec: 232.0; LeNet_lr: 0.00069483527; 
    FastEstimator-Train: step: 800; ce: 0.048837163; steps/sec: 237.98; LeNet_lr: 0.00061537593; 
    FastEstimator-Train: step: 900; ce: 0.06549837; steps/sec: 239.68; LeNet_lr: 0.0005326991; 
    FastEstimator-Train: step: 1000; ce: 0.03539773; steps/sec: 261.57; LeNet_lr: 0.00044912045; 
    FastEstimator-Train: step: 1100; ce: 0.024852607; steps/sec: 242.46; LeNet_lr: 0.00036698082; 
    FastEstimator-Train: step: 1200; ce: 0.009801324; steps/sec: 238.13; LeNet_lr: 0.0002885808; 
    FastEstimator-Train: step: 1300; ce: 0.008669127; steps/sec: 248.62; LeNet_lr: 0.00021611621; 
    FastEstimator-Train: step: 1400; ce: 0.007848155; steps/sec: 244.37; LeNet_lr: 0.00015161661; 
    FastEstimator-Train: step: 1500; ce: 0.05464792; steps/sec: 250.4; LeNet_lr: 9.688851e-05; 
    FastEstimator-Train: step: 1600; ce: 0.02855098; steps/sec: 248.97; LeNet_lr: 5.3464726e-05; 
    FastEstimator-Train: step: 1700; ce: 0.008238091; steps/sec: 254.33; LeNet_lr: 2.2561479e-05; 
    FastEstimator-Train: step: 1800; ce: 0.011979178; steps/sec: 238.57; LeNet_lr: 5.0442964e-06; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 8.09 sec; 
    FastEstimator-Train: step: 1900; ce: 0.18523389; steps/sec: 172.74; LeNet_lr: 0.0009995962; 
    FastEstimator-Train: step: 2000; ce: 0.024744287; steps/sec: 243.07; LeNet_lr: 0.000989258; 
    FastEstimator-Train: step: 2100; ce: 0.010425572; steps/sec: 245.96; LeNet_lr: 0.0009652308; 
    FastEstimator-Train: step: 2200; ce: 0.039544936; steps/sec: 242.05; LeNet_lr: 0.0009281874; 
    FastEstimator-Train: step: 2300; ce: 0.015592085; steps/sec: 244.87; LeNet_lr: 0.00087916537; 
    FastEstimator-Train: step: 2400; ce: 0.19568086; steps/sec: 237.01; LeNet_lr: 0.0008195377; 
    FastEstimator-Train: step: 2500; ce: 0.067701675; steps/sec: 230.62; LeNet_lr: 0.00075097446; 
    FastEstimator-Train: step: 2600; ce: 0.033123072; steps/sec: 244.24; LeNet_lr: 0.00067539595; 
    FastEstimator-Train: step: 2700; ce: 0.012425942; steps/sec: 237.53; LeNet_lr: 0.0005949189; 
    FastEstimator-Train: step: 2800; ce: 0.015756924; steps/sec: 240.78; LeNet_lr: 0.00051179744; 
    FastEstimator-Train: step: 2900; ce: 0.0013934085; steps/sec: 235.39; LeNet_lr: 0.00042835958; 
    FastEstimator-Train: step: 3000; ce: 0.0025910474; steps/sec: 226.27; LeNet_lr: 0.0003469422; 
    FastEstimator-Train: step: 3100; ce: 0.021885639; steps/sec: 236.26; LeNet_lr: 0.00026982563; 
    FastEstimator-Train: step: 3200; ce: 0.030642875; steps/sec: 246.07; LeNet_lr: 0.0001991698; 
    FastEstimator-Train: step: 3300; ce: 0.008929459; steps/sec: 248.81; LeNet_lr: 0.00013695359; 
    FastEstimator-Train: step: 3400; ce: 0.055717546; steps/sec: 228.83; LeNet_lr: 8.491957e-05; 
    FastEstimator-Train: step: 3500; ce: 0.02958621; steps/sec: 233.02; LeNet_lr: 4.452509e-05; 
    FastEstimator-Train: step: 3600; ce: 0.015130471; steps/sec: 246.14; LeNet_lr: 1.6901524e-05; 
    FastEstimator-Train: step: 3700; ce: 0.26439255; steps/sec: 237.31; LeNet_lr: 2.8225472e-06; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 8.01 sec; 
    FastEstimator-Finish: step: 3750; total_time: 16.15 sec; LeNet_lr: 1.0007011e-06; 



```python
visualize_logs(history3, include_metrics="LeNet_lr")
```


![png](assets/tutorial/t07_learning_rate_scheduling_files/t07_learning_rate_scheduling_15_0.png)

