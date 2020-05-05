# Advanced Tutorial 5: Scheduler
In this tutorial, we will talk about:
* **Scheduler:**
    * Concept
    * EpochScheduler
    * RepeatScheduler
* **Things you can schedule:**
    * dataset
    * batch_size
    * NumpyOps
    * optimizer
    * TensorOps
    * Trace

## Scheduler
### Concept
Deep learning training is getting more complicated every year, one major aspect of the complexity is about time-dependent training. For example:
* use different dataset for different training epochs.
* apply different preprocessing for different epochs.
* train different network on different epochs. 
* ...

The list goes on and on, in order to provide easy way for user to accomplish time-dependent training, we provide `Scheduler` class which can help you scheduler any part of the training system. 

Please note that the basic time unit that `Scheduler` can handle `epoch`. If users wants arbitary scheduling cycle, the most trivial way is to customize the length of one epoch in `Estimator`.

### EpochScheduler
The most obvious way to schedule things is through a epoch-value mapping. For example If users want to schedle the batchsize in the following way:

* epoch 1 - batchsize 16
* epoch 2 - batchsize 32
* epoch 3 - batchsize 32
* epoch 4 - batchsize 64
* epoch 5 - batchsize 64

You can do the following:


```python
from fastestimator.schedule import EpochScheduler
batch_size = EpochScheduler(epoch_dict={1:16, 2:32, 4:64})
```


```python
for epoch in range(1, 6):
    print("At epoch {}, batch size is {}".format(epoch, batch_size.get_current_value(epoch)))
```

    At epoch 1, batch size is 16
    At epoch 2, batch size is 32
    At epoch 3, batch size is 32
    At epoch 4, batch size is 64
    At epoch 5, batch size is 64


### RepeatScheduler
If your schedule follows a repeated pattern, then you don't want to specify that for all epochs. `RepeatScheduler` is there to help you. Let's say we want batch size on odd epoch is 32, on even epoch is 64:


```python
from fastestimator.schedule import RepeatScheduler
batch_size = RepeatScheduler(repeat_list=[32, 64])

for epoch in range(1, 6):
    print("At epoch {}, batch size is {}".format(epoch, batch_size.get_current_value(epoch)))
```

    At epoch 1, batch size is 32
    At epoch 2, batch size is 64
    At epoch 3, batch size is 32
    At epoch 4, batch size is 64
    At epoch 5, batch size is 32


## Things you can schedule:

### dataset
Scheduling training or evaluation dataset is very common in deep learning, for example in curriculum learning, people will train on an easy dataset first then gradually move on to harder dataset.  For illustration purpose, let's use two different instance of the same mnist dataset:


```python
from fastestimator.dataset.data import mnist, cifar10
from fastestimator.schedule import EpochScheduler

train_data1, eval_data = mnist.load_data()
train_data2, _ = mnist.load_data()
train_data = EpochScheduler(epoch_dict={1:train_data1, 3: train_data2})
```

### batch size
We can also schedule the batch size on different epochs to make gpu more efficient.


```python
batch_size = RepeatScheduler(repeat_list=[32,64])
```

### NumpyOp
Preprocessing operators can also be scheduled. For illustration purpose, we will apply an `Rotation` for first two epochs and not applying it for the third epoch:


```python
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.numpyop.multivariate import Rotate
import fastestimator as fe

resize_op = EpochScheduler(epoch_dict={1:Rotate(image_in="x", image_out="x",limit=30), 3:None})

pipeline = fe.Pipeline(train_data=train_data, 
                       eval_data=eval_data,
                       batch_size=batch_size, 
                       ops=[ExpandDims(inputs="x", outputs="x"), resize_op, Minmax(inputs="x", outputs="x")])
```

### optimizer
For a fast convergence, some people like to use different optimizer at different phase. In our example,we will use `adam` for the first epoch and `sgd` for the second epoch. 


```python
from fastestimator.architecture.tensorflow import LeNet

model_1 = fe.build(model_fn=LeNet, optimizer_fn=EpochScheduler(epoch_dict={1:"adam", 2: "sgd"}), model_name="m1")
```

### TensorOp
We can schedule TensorOp just like NumpyOp. Let's define another model `model_2` such that:
* epoch 1-2: train `model_1`
* epoch 3: train `model_2`


```python
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.op.tensorop.loss import CrossEntropy

model_2 = fe.build(model_fn=LeNet, optimizer_fn="adam", model_name="m2")

model_map = {1: ModelOp(model=model_1, inputs="x", outputs="y_pred"), 
             3: ModelOp(model=model_2, inputs="x", outputs="y_pred")}

update_map = {1: UpdateOp(model=model_1, loss_name="ce"), 3: UpdateOp(model=model_2, loss_name="ce")}

network = fe.Network(ops=[EpochScheduler(model_map),
                          CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
                          EpochScheduler(update_map)])
```

### Trace
`Trace` can also be scheduled. For example, we will save `model_1` at the end of second epoch and save `model_3` at the end of third epoch:


```python
from fastestimator.trace.io import ModelSaver
import tempfile

save_folder = tempfile.mkdtemp()

#Disable model saving by setting None on 3rd epoch:
modelsaver1 = EpochScheduler({2:ModelSaver(model=model_1,save_dir=save_folder), 3:None})

modelsaver2 = EpochScheduler({3:ModelSaver(model=model_2,save_dir=save_folder)})

traces=[modelsaver1, modelsaver2]
```

## Let the training begin
Nothing special in here, create the estimator then start the training:


```python
estimator = fe.Estimator(pipeline=pipeline, network=network, traces=traces, epochs=3, log_steps=300)
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; m1_lr: 0.01; m2_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.2807064; 
    FastEstimator-Train: step: 300; ce: 0.4982147; steps/sec: 64.01; 
    FastEstimator-Train: step: 600; ce: 0.30764195; steps/sec: 61.3; 
    FastEstimator-Train: step: 900; ce: 0.034835115; steps/sec: 63.88; 
    FastEstimator-Train: step: 1200; ce: 0.118852824; steps/sec: 61.12; 
    FastEstimator-Train: step: 1500; ce: 0.0343146; steps/sec: 63.87; 
    FastEstimator-Train: step: 1800; ce: 0.017417222; steps/sec: 67.63; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 32.13 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.081667416; 
    FastEstimator-Train: step: 2100; ce: 0.0079841465; steps/sec: 43.46; 
    FastEstimator-Train: step: 2400; ce: 0.05044716; steps/sec: 41.7; 
    FastEstimator-Train: step: 2700; ce: 0.038758844; steps/sec: 39.84; 
    FastEstimator-ModelSaver: saved model to /var/folders/5g/d_ny7h211cj3zqkzrtq01s480000gn/T/tmpyju81v09/m1_epoch_2.h5
    FastEstimator-Train: step: 2813; epoch: 2; epoch_time: 23.39 sec; 
    FastEstimator-Eval: step: 2813; epoch: 2; ce: 0.053887773; 
    FastEstimator-Train: step: 3000; ce: 0.1177908; steps/sec: 47.94; 
    FastEstimator-Train: step: 3300; ce: 0.19994278; steps/sec: 65.24; 
    FastEstimator-Train: step: 3600; ce: 0.15240008; steps/sec: 63.68; 
    FastEstimator-Train: step: 3900; ce: 0.018807393; steps/sec: 71.1; 
    FastEstimator-Train: step: 4200; ce: 0.22537899; steps/sec: 66.45; 
    FastEstimator-Train: step: 4500; ce: 0.13255036; steps/sec: 63.33; 
    FastEstimator-ModelSaver: saved model to /var/folders/5g/d_ny7h211cj3zqkzrtq01s480000gn/T/tmpyju81v09/m2_epoch_3.h5
    FastEstimator-Train: step: 4688; epoch: 3; epoch_time: 29.04 sec; 
    FastEstimator-Eval: step: 4688; epoch: 3; ce: 0.05203832; 
    FastEstimator-Finish: step: 4688; total_time: 89.5 sec; m1_lr: 0.01; m2_lr: 0.001; 

