# Advanced Tutorial 5: Scheduler
In this tutorial, we will talk about:
* [Scheduler](./tutorials/r1.1/advanced/t05_scheduler#ta05scheduler)
    * [Concept](./tutorials/r1.1/advanced/t05_scheduler#ta05concept)
    * [EpochScheduler](./tutorials/r1.1/advanced/t05_scheduler#ta05epoch)
    * [RepeatScheduler](./tutorials/r1.1/advanced/t05_scheduler#ta05repeat)
* [Things You Can Schedule](./tutorials/r1.1/advanced/t05_scheduler#ta05things)
    * [Datasets](./tutorials/r1.1/advanced/t05_scheduler#ta05dataset)
    * [Batch Size](./tutorials/r1.1/advanced/t05_scheduler#ta05batch)
    * [NumpyOps](./tutorials/r1.1/advanced/t05_scheduler#ta05numpy)
    * [Optimizers](./tutorials/r1.1/advanced/t05_scheduler#ta05optimizer)
    * [TensorOps](./tutorials/r1.1/advanced/t05_scheduler#ta05tensor)
    * [Traces](./tutorials/r1.1/advanced/t05_scheduler#ta05trace)
* [Related Apphub Examples](./tutorials/r1.1/advanced/t05_scheduler#ta05apphub)

<a id='ta05scheduler'></a>

## Scheduler

<a id='ta05concept'></a>

### Concept
Deep learning training is getting more complicated every year. One major aspect of this complexity is time-dependent training. For example:

* Using different datasets for different training epochs.
* Applying different preprocessing for different epochs.
* Training different networks on different epochs. 
* ...

The list goes on and on. In order to provide an easy way for users to accomplish time-dependent training, we provide the `Scheduler` class which can help you schedule any part of the training. 

Please note that the basic time unit that `Scheduler` can handle is `epochs`. If users want arbitrary scheduling cycles, the simplest way is to customize the length of one epoch in `Estimator` using max_train_steps_per_epoch.

<a id='ta05epoch'></a>

### EpochScheduler
The most straightforward way to schedule things is through an epoch-value mapping. For example, If users want to schedule the batch size in the following way:

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


<a id='ta05repeat'></a>

### RepeatScheduler
If your schedule follows a repeating pattern, then you don't want to specify that for all epochs. `RepeatScheduler` is here to help you. Let's say we want the batch size on odd epochs to be 32, and on even epochs to be 64:


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


<a id='ta05things'></a>

## Things You Can Schedule:

<a id='ta05dataset'></a>

### Datasets
Scheduling training or evaluation datasets is very common in deep learning. For example, in curriculum learning people will train on an easy dataset first and then gradually move on to harder datasets. For illustration purposes, let's use two different instances of the same MNIST dataset:


```python
from fastestimator.dataset.data import mnist, cifar10
from fastestimator.schedule import EpochScheduler

train_data1, eval_data = mnist.load_data()
train_data2, _ = mnist.load_data()
train_data = EpochScheduler(epoch_dict={1:train_data1, 3: train_data2})
```

<a id='ta05batch'></a>

### Batch Size
We can also schedule the batch size on different epochs, which may help resolve GPU resource constraints.


```python
batch_size = RepeatScheduler(repeat_list=[32,64])
```

<a id='ta05numpy'></a>

### NumpyOps
Preprocessing operators can also be scheduled. For illustration purpose, we will apply a `Rotation` for the first two epochs and then not apply it for the third epoch:


```python
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.numpyop.multivariate import Rotate
import fastestimator as fe

rotate_op = EpochScheduler(epoch_dict={1:Rotate(image_in="x", image_out="x",limit=30), 3:None})

pipeline = fe.Pipeline(train_data=train_data, 
                       eval_data=eval_data,
                       batch_size=batch_size, 
                       ops=[ExpandDims(inputs="x", outputs="x"), rotate_op, Minmax(inputs="x", outputs="x")])
```

<a id='ta05optimizer'></a>

### Optimizers
For fast convergence, some people like to use different optimizers at different training phases. In our example, we will use `adam` for the first epoch and `sgd` for the second epoch. 


```python
from fastestimator.architecture.tensorflow import LeNet

model_1 = fe.build(model_fn=LeNet, optimizer_fn=EpochScheduler(epoch_dict={1:"adam", 2: "sgd"}), model_name="m1")
```

<a id='ta05tensor'></a>

### TensorOps
We can schedule `TensorOps` just like `NumpyOps`. Let's define another model `model_2` such that:
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

<a id='ta05trace'></a>

### Traces
`Traces` can also be scheduled. For example, we will save `model_1` at the end of second epoch and save `model_3` at the end of third epoch:


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
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 300; 
    FastEstimator-Train: step: 1; ce: 0.016785031; 
    FastEstimator-Train: step: 300; ce: 0.16430134; steps/sec: 697.26; 
    FastEstimator-Train: step: 600; ce: 0.023913195; steps/sec: 728.45; 
    FastEstimator-Train: step: 900; ce: 0.042380013; steps/sec: 732.24; 
    FastEstimator-Train: step: 1200; ce: 0.0014684915; steps/sec: 723.88; 
    FastEstimator-Train: step: 1500; ce: 0.020901386; steps/sec: 728.1; 
    FastEstimator-Train: step: 1800; ce: 0.0114256; steps/sec: 724.26; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 2.66 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.054206606; 
    FastEstimator-Train: step: 2100; ce: 0.023387551; steps/sec: 546.57; 
    FastEstimator-Train: step: 2400; ce: 0.0030879583; steps/sec: 627.71; 
    FastEstimator-Train: step: 2700; ce: 0.10354612; steps/sec: 631.19; 
    FastEstimator-ModelSaver: Saved model to /tmp/tmph72ava81/m1_epoch_2.h5
    FastEstimator-Train: step: 2813; epoch: 2; epoch_time: 1.58 sec; 
    FastEstimator-Eval: step: 2813; epoch: 2; ce: 0.040080495; 
    FastEstimator-Train: step: 3000; ce: 0.0011174735; steps/sec: 627.96; 
    FastEstimator-Train: step: 3300; ce: 0.019162945; steps/sec: 792.26; 
    FastEstimator-Train: step: 3600; ce: 0.21189407; steps/sec: 796.04; 
    FastEstimator-Train: step: 3900; ce: 0.0007937134; steps/sec: 811.5; 
    FastEstimator-Train: step: 4200; ce: 0.002208311; steps/sec: 818.86; 
    FastEstimator-Train: step: 4500; ce: 0.005765636; steps/sec: 815.32; 
    FastEstimator-ModelSaver: Saved model to /tmp/tmph72ava81/m2_epoch_3.h5
    FastEstimator-Train: step: 4688; epoch: 3; epoch_time: 2.4 sec; 
    FastEstimator-Eval: step: 4688; epoch: 3; ce: 0.033545353; 
    FastEstimator-Finish: step: 4688; total_time: 10.79 sec; m2_lr: 0.001; m1_lr: 0.01; 


<a id='ta05apphub'></a>

## Apphub Examples
You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [PGGAN](./examples/r1.1/image_generation/pggan)
