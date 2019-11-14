# Tutorial 9: Learning Rate Controller
___
In this tutorial, we are going to show you how to use a specific Trace - `LRController` to __change your learning rate during the training__.  In general, `LRController` takes care of both learning rate scheduling as well as changing learning rate on validation.

If you are a keras user, you can see `LRController` as a combination of LRScheduler and ReduceLROnPlateau.


```python
import numpy as np
import tensorflow as tf
import fastestimator as fe
import matplotlib.pyplot as plt
```

## Step 0- Preparation


```python
from fastestimator.architecture import LeNet
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.op.tensorop.loss import SparseCategoricalCrossentropy
from fastestimator.op.tensorop import Minmax

# Create a function to get Pipeline and Network
def get_pipeline_network():
    # step 1. Prepare data
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    train_data = {"x": np.expand_dims(x_train, -1), "y": y_train}
    eval_data = {"x": np.expand_dims(x_eval, -1), "y": y_eval}
    data = {"train": train_data, "eval": eval_data}
    pipeline = fe.Pipeline(batch_size=32, data=data, ops=Minmax(inputs="x", outputs="x"))

    # step 2. Prepare model
    model = fe.build(model_def=LeNet, model_name="lenet", optimizer="adam", loss_name="my_loss")
    network = fe.Network(ops=[ModelOp(inputs="x", model=model, outputs="y_pred"),
                              SparseCategoricalCrossentropy(inputs=("y", "y_pred"), outputs="my_loss")])
    return pipeline, network
```

## Option 1- Customize the learning rate: step-wise control

Let's define our learning rate scheduler to be 0.001 * (1 + step // 500).


```python
from fastestimator.trace import LRController
from fastestimator.schedule import LRSchedule

# Create a LR Scheduler with a custom schedule_fn
class MyLRSchedule1(LRSchedule):
    def schedule_fn(self, current_step_or_epoch, lr):
        lr = 0.001 * (1 + current_step_or_epoch // 500)
        return lr

# Create pipeline, network and lr_scheduler
pipeline1, network1 = get_pipeline_network()
lr_scheduler1 = MyLRSchedule1(schedule_mode="step") # we want to change the lr at a step level

# In Estimator, indicate in traces the LR Scheduler using LR Controller, you also have to specify the model_name.
estimator1 = fe.Estimator(pipeline=pipeline1,
                         network=network1,
                         epochs=2,
                         traces=LRController(model_name="lenet", lr_schedule=lr_scheduler1))
```


```python
# Save the training history and train the model
history1 = estimator1.fit(summary="custom_lr_step")
```

        ______           __  ______     __  _                 __
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/


    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 0; total_train_steps: 3750; lenet_lr: 0.001;
    FastEstimator-Train: step: 0; my_loss: 2.3481772; lenet_lr: 0.001;
    FastEstimator-Train: step: 100; my_loss: 0.2464256; examples/sec: 1857.3; progress: 2.7%; lenet_lr: 0.001;
    FastEstimator-Train: step: 200; my_loss: 0.1358467; examples/sec: 1970.5; progress: 5.3%; lenet_lr: 0.001;
    FastEstimator-Train: step: 300; my_loss: 0.1552827; examples/sec: 2023.7; progress: 8.0%; lenet_lr: 0.001;
    FastEstimator-Train: step: 400; my_loss: 0.0791639; examples/sec: 1910.5; progress: 10.7%; lenet_lr: 0.001;
    FastEstimator-Train: step: 500; my_loss: 0.04834; examples/sec: 1917.2; progress: 13.3%; lenet_lr: 0.002;
    FastEstimator-Train: step: 600; my_loss: 0.3018573; examples/sec: 1983.9; progress: 16.0%; lenet_lr: 0.002;
    FastEstimator-Train: step: 700; my_loss: 0.4213096; examples/sec: 1968.5; progress: 18.7%; lenet_lr: 0.002;
    FastEstimator-Train: step: 800; my_loss: 0.1573652; examples/sec: 1908.6; progress: 21.3%; lenet_lr: 0.002;
    FastEstimator-Train: step: 900; my_loss: 0.0205547; examples/sec: 1830.3; progress: 24.0%; lenet_lr: 0.002;
    FastEstimator-Train: step: 1000; my_loss: 0.0433166; examples/sec: 1845.9; progress: 26.7%; lenet_lr: 0.003;
    FastEstimator-Train: step: 1100; my_loss: 0.0510854; examples/sec: 1817.9; progress: 29.3%; lenet_lr: 0.003;
    FastEstimator-Train: step: 1200; my_loss: 0.0383802; examples/sec: 1920.3; progress: 32.0%; lenet_lr: 0.003;
    FastEstimator-Train: step: 1300; my_loss: 0.0866218; examples/sec: 1767.6; progress: 34.7%; lenet_lr: 0.003;
    FastEstimator-Train: step: 1400; my_loss: 0.0357866; examples/sec: 1795.9; progress: 37.3%; lenet_lr: 0.003;
    FastEstimator-Train: step: 1500; my_loss: 0.156076; examples/sec: 1817.0; progress: 40.0%; lenet_lr: 0.004;
    FastEstimator-Train: step: 1600; my_loss: 0.0259344; examples/sec: 1788.2; progress: 42.7%; lenet_lr: 0.004;
    FastEstimator-Train: step: 1700; my_loss: 0.0738236; examples/sec: 1835.8; progress: 45.3%; lenet_lr: 0.004;
    FastEstimator-Train: step: 1800; my_loss: 0.001702; examples/sec: 1813.8; progress: 48.0%; lenet_lr: 0.004;
    FastEstimator-Eval: step: 1875; epoch: 0; my_loss: 0.0606113; min_my_loss: 0.060611274; since_best_loss: 0;
    FastEstimator-Train: step: 1900; my_loss: 0.1023203; examples/sec: 1649.7; progress: 50.7%; lenet_lr: 0.004;
    FastEstimator-Train: step: 2000; my_loss: 0.0577092; examples/sec: 1698.8; progress: 53.3%; lenet_lr: 0.005;
    FastEstimator-Train: step: 2100; my_loss: 0.0053846; examples/sec: 1700.6; progress: 56.0%; lenet_lr: 0.005;
    FastEstimator-Train: step: 2200; my_loss: 0.1702599; examples/sec: 1712.7; progress: 58.7%; lenet_lr: 0.005;
    FastEstimator-Train: step: 2300; my_loss: 0.0634535; examples/sec: 1638.8; progress: 61.3%; lenet_lr: 0.005;
    FastEstimator-Train: step: 2400; my_loss: 0.1990759; examples/sec: 1634.5; progress: 64.0%; lenet_lr: 0.005;
    FastEstimator-Train: step: 2500; my_loss: 0.01435; examples/sec: 1558.3; progress: 66.7%; lenet_lr: 0.006;
    FastEstimator-Train: step: 2600; my_loss: 0.0282347; examples/sec: 1578.2; progress: 69.3%; lenet_lr: 0.006;
    FastEstimator-Train: step: 2700; my_loss: 0.5667908; examples/sec: 1775.0; progress: 72.0%; lenet_lr: 0.006;
    FastEstimator-Train: step: 2800; my_loss: 0.0600934; examples/sec: 1726.1; progress: 74.7%; lenet_lr: 0.006;
    FastEstimator-Train: step: 2900; my_loss: 0.0223356; examples/sec: 1714.2; progress: 77.3%; lenet_lr: 0.006;
    FastEstimator-Train: step: 3000; my_loss: 0.2202649; examples/sec: 1720.3; progress: 80.0%; lenet_lr: 0.007;
    FastEstimator-Train: step: 3100; my_loss: 0.0588391; examples/sec: 1803.8; progress: 82.7%; lenet_lr: 0.007;
    FastEstimator-Train: step: 3200; my_loss: 0.1268936; examples/sec: 1812.9; progress: 85.3%; lenet_lr: 0.007;
    FastEstimator-Train: step: 3300; my_loss: 0.0441418; examples/sec: 1774.6; progress: 88.0%; lenet_lr: 0.007;
    FastEstimator-Train: step: 3400; my_loss: 0.2001008; examples/sec: 1747.5; progress: 90.7%; lenet_lr: 0.007;
    FastEstimator-Train: step: 3500; my_loss: 0.1163124; examples/sec: 1761.3; progress: 93.3%; lenet_lr: 0.008;
    FastEstimator-Train: step: 3600; my_loss: 0.3343461; examples/sec: 1815.1; progress: 96.0%; lenet_lr: 0.008;
    FastEstimator-Train: step: 3700; my_loss: 0.0862542; examples/sec: 1765.7; progress: 98.7%; lenet_lr: 0.008;
    FastEstimator-Eval: step: 3750; epoch: 1; my_loss: 0.0994106; min_my_loss: 0.060611274; since_best_loss: 1;
    FastEstimator-Finish: step: 3750; total_time: 71.32 sec; lenet_lr: 0.008;


We will use visualize_logs to plot the learning rate. We only want to display this metric, so we specify it in include_metrics.


```python
from fastestimator.summary import visualize_logs

# Show the learning rates history for each step
visualize_logs(history1, include_metrics="lenet_lr")
```


![png](assets/tutorial/t09_learning_rate_controller_files/t09_learning_rate_controller_8_0.png)


## Option 2 - Customize the learning rate: epoch-wise control

Let's now define learning rate to be (epoch +1 ) * 0.002. The only change will be in schedule_mode.


```python
from fastestimator.trace import LRController
from fastestimator.schedule import LRSchedule

# We define our custom Scheduler in the same way as above.
class MyLRSchedule2(LRSchedule):
    def schedule_fn(self, current_step_or_epoch, lr):
        lr = 0.002 * (1 + current_step_or_epoch)
        return lr

# Create pipeline and network.
pipeline2, network2 = get_pipeline_network()

# Here we now indicate epoch as schedule_mode.
lr_scheduler2 = MyLRSchedule2(schedule_mode="epoch")
estimator2 = fe.Estimator(pipeline=pipeline2,
                         network=network2,
                         epochs=2,
                         traces=LRController(model_name="lenet", lr_schedule=lr_scheduler2))
```


```python
# Train and save history
history2 = estimator2.fit(summary="custom_lr_epoch")
```

        ______           __  ______     __  _                 __
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/


    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 0; total_train_steps: 3750; lenet_lr: 0.001;
    FastEstimator-Train: step: 0; my_loss: 2.3128722; lenet_lr: 0.002;
    FastEstimator-Train: step: 100; my_loss: 0.1436051; examples/sec: 1922.5; progress: 2.7%; lenet_lr: 0.002;
    FastEstimator-Train: step: 200; my_loss: 0.1114523; examples/sec: 1913.9; progress: 5.3%; lenet_lr: 0.002;
    FastEstimator-Train: step: 300; my_loss: 0.2748897; examples/sec: 1835.8; progress: 8.0%; lenet_lr: 0.002;
    FastEstimator-Train: step: 400; my_loss: 0.188141; examples/sec: 1725.7; progress: 10.7%; lenet_lr: 0.002;
    FastEstimator-Train: step: 500; my_loss: 0.120022; examples/sec: 1735.0; progress: 13.3%; lenet_lr: 0.002;
    FastEstimator-Train: step: 600; my_loss: 0.1664807; examples/sec: 1744.6; progress: 16.0%; lenet_lr: 0.002;
    FastEstimator-Train: step: 700; my_loss: 0.0196662; examples/sec: 1763.4; progress: 18.7%; lenet_lr: 0.002;
    FastEstimator-Train: step: 800; my_loss: 0.0066346; examples/sec: 1810.4; progress: 21.3%; lenet_lr: 0.002;
    FastEstimator-Train: step: 900; my_loss: 0.0976581; examples/sec: 1800.5; progress: 24.0%; lenet_lr: 0.002;
    FastEstimator-Train: step: 1000; my_loss: 0.0243707; examples/sec: 1811.0; progress: 26.7%; lenet_lr: 0.002;
    FastEstimator-Train: step: 1100; my_loss: 0.0514063; examples/sec: 1788.3; progress: 29.3%; lenet_lr: 0.002;
    FastEstimator-Train: step: 1200; my_loss: 0.0330723; examples/sec: 1732.4; progress: 32.0%; lenet_lr: 0.002;
    FastEstimator-Train: step: 1300; my_loss: 0.0751199; examples/sec: 1750.7; progress: 34.7%; lenet_lr: 0.002;
    FastEstimator-Train: step: 1400; my_loss: 0.0199403; examples/sec: 1742.0; progress: 37.3%; lenet_lr: 0.002;
    FastEstimator-Train: step: 1500; my_loss: 0.0034511; examples/sec: 1749.6; progress: 40.0%; lenet_lr: 0.002;
    FastEstimator-Train: step: 1600; my_loss: 0.0096388; examples/sec: 1751.2; progress: 42.7%; lenet_lr: 0.002;
    FastEstimator-Train: step: 1700; my_loss: 0.0504964; examples/sec: 1738.3; progress: 45.3%; lenet_lr: 0.002;
    FastEstimator-Train: step: 1800; my_loss: 0.0356065; examples/sec: 1742.5; progress: 48.0%; lenet_lr: 0.002;
    FastEstimator-Eval: step: 1875; epoch: 0; my_loss: 0.04598; min_my_loss: 0.04597995; since_best_loss: 0;
    FastEstimator-Train: step: 1900; my_loss: 0.1542965; examples/sec: 1713.9; progress: 50.7%; lenet_lr: 0.004;
    FastEstimator-Train: step: 2000; my_loss: 0.0607498; examples/sec: 1812.1; progress: 53.3%; lenet_lr: 0.004;
    FastEstimator-Train: step: 2100; my_loss: 0.1583865; examples/sec: 1758.9; progress: 56.0%; lenet_lr: 0.004;
    FastEstimator-Train: step: 2200; my_loss: 0.0503638; examples/sec: 1724.8; progress: 58.7%; lenet_lr: 0.004;
    FastEstimator-Train: step: 2300; my_loss: 0.0146242; examples/sec: 1764.7; progress: 61.3%; lenet_lr: 0.004;
    FastEstimator-Train: step: 2400; my_loss: 0.1219713; examples/sec: 1759.6; progress: 64.0%; lenet_lr: 0.004;
    FastEstimator-Train: step: 2500; my_loss: 0.0502973; examples/sec: 1769.9; progress: 66.7%; lenet_lr: 0.004;
    FastEstimator-Train: step: 2600; my_loss: 0.0020814; examples/sec: 1786.2; progress: 69.3%; lenet_lr: 0.004;
    FastEstimator-Train: step: 2700; my_loss: 0.0083729; examples/sec: 1734.1; progress: 72.0%; lenet_lr: 0.004;
    FastEstimator-Train: step: 2800; my_loss: 0.0129272; examples/sec: 1775.2; progress: 74.7%; lenet_lr: 0.004;
    FastEstimator-Train: step: 2900; my_loss: 0.1544187; examples/sec: 1787.9; progress: 77.3%; lenet_lr: 0.004;
    FastEstimator-Train: step: 3000; my_loss: 0.0066857; examples/sec: 1758.8; progress: 80.0%; lenet_lr: 0.004;
    FastEstimator-Train: step: 3100; my_loss: 0.0012132; examples/sec: 1751.4; progress: 82.7%; lenet_lr: 0.004;
    FastEstimator-Train: step: 3200; my_loss: 0.0240937; examples/sec: 1729.5; progress: 85.3%; lenet_lr: 0.004;
    FastEstimator-Train: step: 3300; my_loss: 0.0055601; examples/sec: 1750.6; progress: 88.0%; lenet_lr: 0.004;
    FastEstimator-Train: step: 3400; my_loss: 0.1114296; examples/sec: 1760.2; progress: 90.7%; lenet_lr: 0.004;
    FastEstimator-Train: step: 3500; my_loss: 0.0691303; examples/sec: 1763.6; progress: 93.3%; lenet_lr: 0.004;
    FastEstimator-Train: step: 3600; my_loss: 0.064943; examples/sec: 1758.0; progress: 96.0%; lenet_lr: 0.004;
    FastEstimator-Train: step: 3700; my_loss: 0.028415; examples/sec: 1646.9; progress: 98.7%; lenet_lr: 0.004;
    FastEstimator-Eval: step: 3750; epoch: 1; my_loss: 0.0435661; min_my_loss: 0.043566078; since_best_loss: 0;
    FastEstimator-Finish: step: 3750; total_time: 70.96 sec; lenet_lr: 0.004;



```python
# Show the learning rate for each step: it changes only at an epoch level!
visualize_logs(history2, include_metrics="lenet_lr")
```


![png](t09_learning_rate_controller_files/t09_learning_rate_controller_12_0.png)


## Option 3- Built-in Cyclic Learning Rate - example 1

FastEstimator provides many pre-implemented popular learning rate shedulers for users.
Here, we are going to use `CyclicLRSchedule`: it is a generalization of many learning rate schedulers.

In the next example, let's decay the learning rate by half of cosine curve.


```python
from fastestimator.trace import LRController
from fastestimator.schedule import CyclicLRSchedule

# Create pipeline and network
pipeline3, network3 = get_pipeline_network()

# Directly use the pre-built CyclicLRSchedule, with a cosine decrease method and one cycle.
estimator3 = fe.Estimator(pipeline=pipeline3,
                         network=network3,
                         epochs=2,
                         traces=LRController(model_name="lenet",
                                             lr_schedule=CyclicLRSchedule(num_cycle=1, decrease_method="cosine")))
```


```python
# Train and save history
history3 = estimator3.fit(summary="cyclic_1")
```

        ______           __  ______     __  _                 __
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/


    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 0; total_train_steps: 3750; lenet_lr: 0.001;
    FastEstimator-Train: step: 0; my_loss: 2.2814844; lenet_lr: 0.001;
    FastEstimator-Train: step: 100; my_loss: 0.2648424; examples/sec: 1909.4; progress: 2.7%; lenet_lr: 0.000998;
    FastEstimator-Train: step: 200; my_loss: 0.2661002; examples/sec: 1845.4; progress: 5.3%; lenet_lr: 0.000993;
    FastEstimator-Train: step: 300; my_loss: 0.0546548; examples/sec: 1801.1; progress: 8.0%; lenet_lr: 0.000984;
    FastEstimator-Train: step: 400; my_loss: 0.4984356; examples/sec: 1814.1; progress: 10.7%; lenet_lr: 0.000972;
    FastEstimator-Train: step: 500; my_loss: 0.2745714; examples/sec: 1738.1; progress: 13.3%; lenet_lr: 0.000957;
    FastEstimator-Train: step: 600; my_loss: 0.2907083; examples/sec: 1740.5; progress: 16.0%; lenet_lr: 0.000938;
    FastEstimator-Train: step: 700; my_loss: 0.2194481; examples/sec: 1747.6; progress: 18.7%; lenet_lr: 0.000917;
    FastEstimator-Train: step: 800; my_loss: 0.0251494; examples/sec: 1723.6; progress: 21.3%; lenet_lr: 0.000892;
    FastEstimator-Train: step: 900; my_loss: 0.0344817; examples/sec: 1755.4; progress: 24.0%; lenet_lr: 0.000865;
    FastEstimator-Train: step: 1000; my_loss: 0.0822155; examples/sec: 1718.3; progress: 26.7%; lenet_lr: 0.000835;
    FastEstimator-Train: step: 1100; my_loss: 0.0764152; examples/sec: 1742.5; progress: 29.3%; lenet_lr: 0.000802;
    FastEstimator-Train: step: 1200; my_loss: 0.1622209; examples/sec: 1779.3; progress: 32.0%; lenet_lr: 0.000768;
    FastEstimator-Train: step: 1300; my_loss: 0.0091959; examples/sec: 1776.9; progress: 34.7%; lenet_lr: 0.000732;
    FastEstimator-Train: step: 1400; my_loss: 0.1187094; examples/sec: 1802.9; progress: 37.3%; lenet_lr: 0.000694;
    FastEstimator-Train: step: 1500; my_loss: 0.0149192; examples/sec: 1822.0; progress: 40.0%; lenet_lr: 0.000655;
    FastEstimator-Train: step: 1600; my_loss: 0.1533182; examples/sec: 1794.1; progress: 42.7%; lenet_lr: 0.000615;
    FastEstimator-Train: step: 1700; my_loss: 0.0039344; examples/sec: 1756.7; progress: 45.3%; lenet_lr: 0.000573;
    FastEstimator-Train: step: 1800; my_loss: 0.018994; examples/sec: 1741.9; progress: 48.0%; lenet_lr: 0.000532;
    FastEstimator-Eval: step: 1875; epoch: 0; my_loss: 0.0433684; min_my_loss: 0.043368418; since_best_loss: 0;
    FastEstimator-Train: step: 1900; my_loss: 0.0032261; examples/sec: 1666.2; progress: 50.7%; lenet_lr: 0.00049;
    FastEstimator-Train: step: 2000; my_loss: 0.0739067; examples/sec: 1789.4; progress: 53.3%; lenet_lr: 0.000448;
    FastEstimator-Train: step: 2100; my_loss: 0.0028001; examples/sec: 1806.5; progress: 56.0%; lenet_lr: 0.000407;
    FastEstimator-Train: step: 2200; my_loss: 0.0442757; examples/sec: 1776.1; progress: 58.7%; lenet_lr: 0.000366;
    FastEstimator-Train: step: 2300; my_loss: 0.0145678; examples/sec: 1753.7; progress: 61.3%; lenet_lr: 0.000326;
    FastEstimator-Train: step: 2400; my_loss: 0.0232146; examples/sec: 1767.4; progress: 64.0%; lenet_lr: 0.000288;
    FastEstimator-Train: step: 2500; my_loss: 0.1448252; examples/sec: 1806.0; progress: 66.7%; lenet_lr: 0.000251;
    FastEstimator-Train: step: 2600; my_loss: 0.0821529; examples/sec: 1803.9; progress: 69.3%; lenet_lr: 0.000215;
    FastEstimator-Train: step: 2700; my_loss: 0.1417835; examples/sec: 1821.3; progress: 72.0%; lenet_lr: 0.000182;
    FastEstimator-Train: step: 2800; my_loss: 0.0585024; examples/sec: 1828.4; progress: 74.7%; lenet_lr: 0.000151;
    FastEstimator-Train: step: 2900; my_loss: 0.0156975; examples/sec: 1806.4; progress: 77.3%; lenet_lr: 0.000122;
    FastEstimator-Train: step: 3000; my_loss: 0.0738409; examples/sec: 1831.4; progress: 80.0%; lenet_lr: 9.6e-05;
    FastEstimator-Train: step: 3100; my_loss: 0.028531; examples/sec: 1833.8; progress: 82.7%; lenet_lr: 7.3e-05;
    FastEstimator-Train: step: 3200; my_loss: 0.0030303; examples/sec: 1805.4; progress: 85.3%; lenet_lr: 5.3e-05;
    FastEstimator-Train: step: 3300; my_loss: 0.0253246; examples/sec: 1832.6; progress: 88.0%; lenet_lr: 3.6e-05;
    FastEstimator-Train: step: 3400; my_loss: 0.0122934; examples/sec: 1830.3; progress: 90.7%; lenet_lr: 2.2e-05;
    FastEstimator-Train: step: 3500; my_loss: 0.008961; examples/sec: 1780.8; progress: 93.3%; lenet_lr: 1.2e-05;
    FastEstimator-Train: step: 3600; my_loss: 0.0109423; examples/sec: 1767.3; progress: 96.0%; lenet_lr: 5e-06;
    FastEstimator-Train: step: 3700; my_loss: 0.0184381; examples/sec: 1786.7; progress: 98.7%; lenet_lr: 1e-06;
    FastEstimator-Eval: step: 3750; epoch: 1; my_loss: 0.0275551; min_my_loss: 0.027555123; since_best_loss: 0;
    FastEstimator-Finish: step: 3750; total_time: 70.09 sec; lenet_lr: 1e-06;



```python
# Plot the learning rate for each step
visualize_logs(history3, include_metrics="lenet_lr")
```


![png](t09_learning_rate_controller_files/t09_learning_rate_controller_16_0.png)


## Option 3- Built-in Cyclic Learning Rate: example 2

Users can also choose to add more cycles of the learning rate with `num_cycle`, and to change the subsequent cycle length using `cycle_multiplier`. If `cycle_multiplier=2`, the second cycle will be twice as long as the first one.


```python
from fastestimator.trace import LRController
from fastestimator.schedule import CyclicLRSchedule

# We create pipeline and network.
pipeline4, network4 = get_pipeline_network()

# We specify num_cycle and cycle_multiplier in CyclicLRSchedule.
estimator4 = fe.Estimator(pipeline=pipeline4,
                         network=network4,
                         epochs=2,
                         traces=LRController(model_name="lenet",
                                             lr_schedule=CyclicLRSchedule(num_cycle=2,
                                                                          cycle_multiplier=2,
                                                                          decrease_method="cosine")))
```


```python
# Train and save history
history4 = estimator4.fit(summary="cyclic_2")
```

        ______           __  ______     __  _                 __
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/


    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 0; total_train_steps: 3750; lenet_lr: 0.001;
    FastEstimator-Train: step: 0; my_loss: 2.3045583; lenet_lr: 0.001;
    FastEstimator-Train: step: 100; my_loss: 0.464899; examples/sec: 1934.2; progress: 2.7%; lenet_lr: 0.000984;
    FastEstimator-Train: step: 200; my_loss: 0.2427881; examples/sec: 1871.5; progress: 5.3%; lenet_lr: 0.000938;
    FastEstimator-Train: step: 300; my_loss: 0.103807; examples/sec: 1834.4; progress: 8.0%; lenet_lr: 0.000865;
    FastEstimator-Train: step: 400; my_loss: 0.140082; examples/sec: 1780.1; progress: 10.7%; lenet_lr: 0.000768;
    FastEstimator-Train: step: 500; my_loss: 0.0394129; examples/sec: 1796.5; progress: 13.3%; lenet_lr: 0.000655;
    FastEstimator-Train: step: 600; my_loss: 0.0523208; examples/sec: 1805.0; progress: 16.0%; lenet_lr: 0.000532;
    FastEstimator-Train: step: 700; my_loss: 0.1014733; examples/sec: 1780.7; progress: 18.7%; lenet_lr: 0.000407;
    FastEstimator-Train: step: 800; my_loss: 0.1789927; examples/sec: 1793.5; progress: 21.3%; lenet_lr: 0.000288;
    FastEstimator-Train: step: 900; my_loss: 0.0215995; examples/sec: 1798.5; progress: 24.0%; lenet_lr: 0.000182;
    FastEstimator-Train: step: 1000; my_loss: 0.1063223; examples/sec: 1752.1; progress: 26.7%; lenet_lr: 9.6e-05;
    FastEstimator-Train: step: 1100; my_loss: 0.0317446; examples/sec: 1791.9; progress: 29.3%; lenet_lr: 3.6e-05;
    FastEstimator-Train: step: 1200; my_loss: 0.0145848; examples/sec: 1800.8; progress: 32.0%; lenet_lr: 5e-06;
    FastEstimator-Train: step: 1300; my_loss: 0.0921576; examples/sec: 1774.9; progress: 34.7%; lenet_lr: 0.000999;
    FastEstimator-Train: step: 1400; my_loss: 0.027713; examples/sec: 1798.2; progress: 37.3%; lenet_lr: 0.000991;
    FastEstimator-Train: step: 1500; my_loss: 0.0133265; examples/sec: 1782.5; progress: 40.0%; lenet_lr: 0.000976;
    FastEstimator-Train: step: 1600; my_loss: 0.0638937; examples/sec: 1796.9; progress: 42.7%; lenet_lr: 0.000952;
    FastEstimator-Train: step: 1700; my_loss: 0.2630475; examples/sec: 1736.1; progress: 45.3%; lenet_lr: 0.000922;
    FastEstimator-Train: step: 1800; my_loss: 0.0194361; examples/sec: 1719.0; progress: 48.0%; lenet_lr: 0.000885;
    FastEstimator-Eval: step: 1875; epoch: 0; my_loss: 0.0710518; min_my_loss: 0.07105176; since_best_loss: 0;
    FastEstimator-Train: step: 1900; my_loss: 0.0496064; examples/sec: 1654.0; progress: 50.7%; lenet_lr: 0.000842;
    FastEstimator-Train: step: 2000; my_loss: 0.1158404; examples/sec: 1802.9; progress: 53.3%; lenet_lr: 0.000794;
    FastEstimator-Train: step: 2100; my_loss: 0.0013254; examples/sec: 1833.2; progress: 56.0%; lenet_lr: 0.000741;
    FastEstimator-Train: step: 2200; my_loss: 0.0100821; examples/sec: 1844.8; progress: 58.7%; lenet_lr: 0.000684;
    FastEstimator-Train: step: 2300; my_loss: 0.0083895; examples/sec: 1831.4; progress: 61.3%; lenet_lr: 0.000625;
    FastEstimator-Train: step: 2400; my_loss: 0.0158134; examples/sec: 1836.7; progress: 64.0%; lenet_lr: 0.000563;
    FastEstimator-Train: step: 2500; my_loss: 0.0190931; examples/sec: 1858.1; progress: 66.7%; lenet_lr: 0.000501;
    FastEstimator-Train: step: 2600; my_loss: 0.0095861; examples/sec: 1881.2; progress: 69.3%; lenet_lr: 0.000438;
    FastEstimator-Train: step: 2700; my_loss: 0.0872287; examples/sec: 1860.3; progress: 72.0%; lenet_lr: 0.000376;
    FastEstimator-Train: step: 2800; my_loss: 0.0044112; examples/sec: 1829.0; progress: 74.7%; lenet_lr: 0.000317;
    FastEstimator-Train: step: 2900; my_loss: 0.0053059; examples/sec: 1828.6; progress: 77.3%; lenet_lr: 0.00026;
    FastEstimator-Train: step: 3000; my_loss: 0.0915491; examples/sec: 1834.0; progress: 80.0%; lenet_lr: 0.000207;
    FastEstimator-Train: step: 3100; my_loss: 0.0111443; examples/sec: 1830.7; progress: 82.7%; lenet_lr: 0.000159;
    FastEstimator-Train: step: 3200; my_loss: 0.0313221; examples/sec: 1799.2; progress: 85.3%; lenet_lr: 0.000116;
    FastEstimator-Train: step: 3300; my_loss: 0.0339224; examples/sec: 1811.1; progress: 88.0%; lenet_lr: 7.9e-05;
    FastEstimator-Train: step: 3400; my_loss: 0.1087273; examples/sec: 1808.4; progress: 90.7%; lenet_lr: 4.9e-05;
    FastEstimator-Train: step: 3500; my_loss: 0.0065629; examples/sec: 1784.6; progress: 93.3%; lenet_lr: 2.5e-05;
    FastEstimator-Train: step: 3600; my_loss: 0.0080364; examples/sec: 1790.1; progress: 96.0%; lenet_lr: 1e-05;
    FastEstimator-Train: step: 3700; my_loss: 0.0077254; examples/sec: 1819.7; progress: 98.7%; lenet_lr: 2e-06;
    FastEstimator-Eval: step: 3750; epoch: 1; my_loss: 0.0301187; min_my_loss: 0.030118655; since_best_loss: 0;
    FastEstimator-Finish: step: 3750; total_time: 69.28 sec; lenet_lr: 1e-06;



```python
# Plot the learning rate
visualize_logs(history4, include_metrics="lenet_lr")
```


![png](t09_learning_rate_controller_files/t09_learning_rate_controller_20_0.png)


We observe that we have two cycles of decreasing learning rate, and the second one is twice as long as the first one.
