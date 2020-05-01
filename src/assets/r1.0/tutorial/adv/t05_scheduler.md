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

for epoch in range(1, 6):
    print("At epoch {}, batch size is {}".format(epoch, batch_size.get_current_value(epoch)))
```

    /anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


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
Scheduling training or evaluation dataset is very common in deep learning, for example in curriculum learning, people will train on an easy task first then gradually move on to harder dataset.  For illustration purpose, let's use two different instance of the same mnist dataset:


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
Preprocessing operators can also be scheduled. For illustration purpose, we will apply an `Resize` for first two epochs and not applying it for the third epoch:


```python
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.numpyop.multivariate import Resize
import fastestimator as fe

resize_op = EpochScheduler(epoch_dict={1:Resize(height=28, width=28, image_in="x", image_out="x"), 3:None})

pipeline = fe.Pipeline(train_data=train_data, 
                       eval_data=eval_data,
                       batch_size=batch_size, 
                       ops=[ExpandDims(inputs="x", outputs="x"), resize_op, Minmax(inputs="x", outputs="x")])
```

### optimizer
For a fast convergence, some people like to use different optimizer at different phase. In our example,we will use `adam` for the first epoch and `sgd` for the second epoch. 


```python
from fastestimator.architecture.tensorflow import LeNet

model_1 = fe.build(model_fn=LeNet, optimizer_fn=EpochScheduler(epoch_dict={1:"adam", 2: "sgd"}))
```

### TensorOp
We can schedule TensorOp just like NumpyOp. Let's define another model `model_2` such that:
* epoch 1-2: train `model_1`
* epoch 3: train `model_2`


```python
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.op.tensorop.loss import CrossEntropy

model_2 = fe.build(model_fn=LeNet, optimizer_fn="adam")

model_map = {1: ModelOp(model=model_1, inputs="x", outputs="y_pred"), 
             3: ModelOp(model=model_2, inputs="x", outputs="y_pred")}

update_map = {1: UpdateOp(model=model_1, loss_name="ce"), 3: UpdateOp(model=model_2, loss_name="ce")}

network = fe.Network(ops=[EpochScheduler(model_map),
                          CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
                          EpochScheduler(update_map)])
```

## Let the training begin
Nothing special in here, create the estimator then start the training:


```python
estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=3, log_steps=300)
estimator.fit()
```

    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; model1_lr: 0.001; model_lr: 0.01; 
    FastEstimator-Train: step: 1; ce: 2.2982383; 
    FastEstimator-Train: step: 300; ce: 0.26133412; steps/sec: 70.14; 
    FastEstimator-Train: step: 600; ce: 0.031249696; steps/sec: 68.53; 
    FastEstimator-Train: step: 900; ce: 0.1134463; steps/sec: 68.67; 
    FastEstimator-Train: step: 1200; ce: 0.083426505; steps/sec: 68.53; 
    FastEstimator-Train: step: 1500; ce: 0.032718502; steps/sec: 66.58; 
    FastEstimator-Train: step: 1800; ce: 0.034739777; steps/sec: 69.45; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 29.85 sec; 
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.057982765; min_ce: 0.057982765; since_best: 0; 
    FastEstimator-Train: step: 2100; ce: 0.012392752; steps/sec: 47.63; 
    FastEstimator-Train: step: 2400; ce: 0.023588017; steps/sec: 43.43; 
    FastEstimator-Train: step: 2700; ce: 0.009507541; steps/sec: 43.52; 
    FastEstimator-Train: step: 2813; epoch: 2; epoch_time: 21.68 sec; 
    FastEstimator-Eval: step: 2813; epoch: 2; ce: 0.033888463; min_ce: 0.033888463; since_best: 0; 
    FastEstimator-Train: step: 3000; ce: 0.11985014; steps/sec: 54.07; 
    FastEstimator-Train: step: 3300; ce: 0.031535544; steps/sec: 70.04; 
    FastEstimator-Train: step: 3600; ce: 0.09138005; steps/sec: 69.48; 
    FastEstimator-Train: step: 3900; ce: 0.06577636; steps/sec: 68.13; 
    FastEstimator-Train: step: 4200; ce: 0.000930603; steps/sec: 68.8; 
    FastEstimator-Train: step: 4500; ce: 0.0047831526; steps/sec: 71.04; 
    FastEstimator-Train: step: 4688; epoch: 3; epoch_time: 27.21 sec; 
    FastEstimator-Eval: step: 4688; epoch: 3; ce: 0.041508634; min_ce: 0.033888463; since_best: 1; 
    FastEstimator-Finish: step: 4688; total_time: 82.15 sec; model1_lr: 0.001; model_lr: 0.01; 

