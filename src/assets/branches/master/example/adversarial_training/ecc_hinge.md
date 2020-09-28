
# Adversarial Robustness with Error Correcting Codes (and Hinge Loss)
## (Never use Softmax again)

In this example we will show how using error correcting codes as a model output can drastically reduce model overfitting while simultaneously increasing model robustness against adversarial attacks. In other words, why you should never use a softmax layer again. This is slightly more complicated than the our other [ECC](../ecc/ecc.ipynb) apphub example, but it allows for more accurate final probability estimates (the FE Hadamard network layer results in probability smoothing which prevents the network from ever being 100% confident in a class choice). The usefulness of error correcting codes was first publicized by the US Army in a [2019 Neurips Paper](https://papers.nips.cc/paper/9070-error-correcting-output-codes-improve-probability-estimation-and-adversarial-robustness-of-deep-neural-networks.pdf). For background on adversarial attacks, and on the attack type we will be demonstrating here, check out our [FGSM](../fgsm/fgsm.ipynb) apphub example. Note that in this apphub we will not be training against adversarial samples, but only performing adversarial attacks during evaluation to see how different models fair against them.

## Imports


```python
import math
import tempfile

from tensorflow.python.keras import Sequential, layers
from tensorflow.python.keras.layers import Concatenate, Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.python.keras.models import Model

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import cifar10
from fastestimator.layers.tensorflow import HadamardCode
from fastestimator.op.numpyop.univariate import Hadamard, Normalize
from fastestimator.op.tensorop import UnHadamard
from fastestimator.op.tensorop.gradient import FGSM, Watch
from fastestimator.op.tensorop.loss import CrossEntropy, Hinge
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.summary.logs import visualize_logs
from fastestimator.trace.adapt import EarlyStopping
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
```


```python
# training parameters
epsilon=0.04  # The strength of the adversarial attack
epochs=60
batch_size=50
log_steps=500
max_train_steps_per_epoch=None
max_eval_steps_per_epoch=None
save_dir=tempfile.mkdtemp()
```

## Getting the Data
For these experiments we will be using the CIFAR-10 Dataset


```python
train_data, eval_data = cifar10.load_data()
test_data = eval_data.split(0.5)
pipeline = fe.Pipeline(
    train_data=train_data,
    eval_data=eval_data,
    test_data=test_data,
    batch_size=batch_size,
    ops=[
        Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        Hadamard(inputs="y", outputs="y_code", n_classes=10)
        ])
```

## Defining Estimators
In this apphub we will be comparing a baseline model against two models using hinge loss to enable training with error correcting codes. The setting up the hinge loss models requires a few extra Ops along the way.


```python
def get_baseline_estimator(model):
    network = fe.Network(ops=[
        Watch(inputs="x", mode=('eval', 'test')),
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="base_ce"),
        UpdateOp(model=model, loss_name="base_ce"),
        FGSM(data="x", loss="base_ce", outputs="x_adverse", epsilon=epsilon, mode=('eval', 'test')),
        ModelOp(model=model, inputs="x_adverse", outputs="y_pred_adv", mode=('eval', 'test')),
        CrossEntropy(inputs=("y_pred_adv", "y"), outputs="adv_ce", mode=('eval', 'test'))
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="base_accuracy"),
        Accuracy(true_key="y", pred_key="y_pred_adv", output_name="adversarial_accuracy"),
        BestModelSaver(model=model, save_dir=save_dir, metric="base_ce", save_best_mode="min", load_best_final=True),
        EarlyStopping(monitor="base_ce", patience=10)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             log_steps=log_steps,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch,
                             monitor_names=["adv_ce"])
    return estimator
```


```python
def get_hinge_estimator(model):
    network = fe.Network(ops=[
        Watch(inputs="x", mode=('eval', 'test')),
        ModelOp(model=model, inputs="x", outputs="y_pred_code"),
        Hinge(inputs=("y_pred_code", "y_code"), outputs="base_hinge"),
        UpdateOp(model=model, loss_name="base_hinge"),
        UnHadamard(inputs="y_pred_code", outputs="y_pred", n_classes=10, mode=('eval', 'test')),
        # The adversarial attack:
        FGSM(data="x", loss="base_hinge", outputs="x_adverse_hinge", epsilon=epsilon, mode=('eval', 'test')),
        ModelOp(model=model, inputs="x_adverse_hinge", outputs="y_pred_adv_hinge_code", mode=('eval', 'test')),
        Hinge(inputs=("y_pred_adv_hinge_code", "y_code"), outputs="adv_hinge", mode=('eval', 'test')),
        UnHadamard(inputs="y_pred_adv_hinge_code", outputs="y_pred_adv_hinge", n_classes=10, mode=('eval', 'test')),
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="base_accuracy"),
        Accuracy(true_key="y", pred_key="y_pred_adv_hinge", output_name="adversarial_accuracy"),
        BestModelSaver(model=model, save_dir=save_dir, metric="base_hinge", save_best_mode="min", load_best_final=True),
        EarlyStopping(monitor="base_hinge", patience=10)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             log_steps=log_steps,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch,
                             monitor_names=["adv_hinge"])
    return estimator
```

## The Models
### 1 - A LeNet model with Softmax


```python
softmax_model = fe.build(model_fn=lambda:LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam", model_name='softmax')
```

### 2 - A LeNet model with Error Correcting Codes


```python
def EccLeNet(input_shape=(32, 32, 3), code_length=16):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(code_length, activation='tanh'))  # Note that this is the only difference between this model and the FE LeNet implementation
    return model
```


```python
ecc_model = fe.build(model_fn=EccLeNet, optimizer_fn="adam", model_name='ecc')
```

### 3 - A LeNet model using ECC and multiple feature heads
While it is common practice to follow the feature extraction layers of convolution networks with several fully connected layers in order to perform classification, this can lead to the final logits being interdependent which can actually reduce the robustness of the network. One way around this is to divide your classification layers into multiple smaller independent units:


```python
def HydraEccLeNet(input_shape=(32, 32, 3), code_length=16):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu')(pool2)
    flat = Flatten()(conv3)
    # Create multiple heads
    n_heads = 4
    heads = [Dense(16, activation='relu')(flat) for _ in range(n_heads)]
    heads2 = [Dense(code_length // n_heads, activation='tanh')(head) for head in heads]
    outputs = Concatenate()(heads2)
    return Model(inputs=inputs, outputs=outputs)
```


```python
hydra_model = fe.build(model_fn=HydraEccLeNet, optimizer_fn="adam", model_name='hydra_ecc')
```

## The Experiments
Let's get Estimators for each of these models and compare them:


```python
softmax_estimator = get_baseline_estimator(softmax_model)
ecc_estimator = get_hinge_estimator(ecc_model)
hydra_estimator = get_hinge_estimator(hydra_model)
```


```python
softmax_estimator.fit('Softmax')
softmax_results = softmax_estimator.test()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 500; 
    FastEstimator-Train: step: 1; base_ce: 2.2913797; 
    FastEstimator-Train: step: 500; base_ce: 1.129351; steps/sec: 80.35; 
    FastEstimator-Train: step: 1000; base_ce: 0.9674388; steps/sec: 66.16; 
    FastEstimator-Train: step: 1000; epoch: 1; epoch_time: 14.46 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 1000; epoch: 1; adv_ce: 2.0666945; base_ce: 1.1314628; base_accuracy: 0.5976; adversarial_accuracy: 0.2934; since_best_base_ce: 0; min_base_ce: 1.1314628; 
    FastEstimator-Train: step: 1500; base_ce: 0.7620763; steps/sec: 88.76; 
    FastEstimator-Train: step: 2000; base_ce: 1.2483466; steps/sec: 80.25; 
    FastEstimator-Train: step: 2000; epoch: 2; epoch_time: 11.87 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 2000; epoch: 2; adv_ce: 2.1358154; base_ce: 0.9702002; base_accuracy: 0.6556; adversarial_accuracy: 0.2818; since_best_base_ce: 0; min_base_ce: 0.9702002; 
    FastEstimator-Train: step: 2500; base_ce: 0.6228955; steps/sec: 82.13; 
    FastEstimator-Train: step: 3000; base_ce: 0.85292035; steps/sec: 69.24; 
    FastEstimator-Train: step: 3000; epoch: 3; epoch_time: 13.31 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 3000; epoch: 3; adv_ce: 2.2940953; base_ce: 0.8567269; base_accuracy: 0.7016; adversarial_accuracy: 0.2634; since_best_base_ce: 0; min_base_ce: 0.8567269; 
    FastEstimator-Train: step: 3500; base_ce: 1.0757314; steps/sec: 67.03; 
    FastEstimator-Train: step: 4000; base_ce: 0.8255354; steps/sec: 64.26; 
    FastEstimator-Train: step: 4000; epoch: 4; epoch_time: 15.24 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 4000; epoch: 4; adv_ce: 2.4554718; base_ce: 0.81987983; base_accuracy: 0.7228; adversarial_accuracy: 0.2714; since_best_base_ce: 0; min_base_ce: 0.81987983; 
    FastEstimator-Train: step: 4500; base_ce: 0.5227963; steps/sec: 61.48; 
    FastEstimator-Train: step: 5000; base_ce: 0.6290733; steps/sec: 62.42; 
    FastEstimator-Train: step: 5000; epoch: 5; epoch_time: 16.14 sec; 
    FastEstimator-Eval: step: 5000; epoch: 5; adv_ce: 2.7408102; base_ce: 0.82819456; base_accuracy: 0.7202; adversarial_accuracy: 0.2584; since_best_base_ce: 1; min_base_ce: 0.81987983; 
    FastEstimator-Train: step: 5500; base_ce: 0.46184003; steps/sec: 60.35; 
    FastEstimator-Train: step: 6000; base_ce: 0.74710757; steps/sec: 63.8; 
    FastEstimator-Train: step: 6000; epoch: 6; epoch_time: 16.13 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 6000; epoch: 6; adv_ce: 3.0637393; base_ce: 0.8165515; base_accuracy: 0.7276; adversarial_accuracy: 0.2316; since_best_base_ce: 0; min_base_ce: 0.8165515; 
    FastEstimator-Train: step: 6500; base_ce: 0.6648202; steps/sec: 59.67; 
    FastEstimator-Train: step: 7000; base_ce: 0.5298349; steps/sec: 63.3; 
    FastEstimator-Train: step: 7000; epoch: 7; epoch_time: 16.28 sec; 
    FastEstimator-Eval: step: 7000; epoch: 7; adv_ce: 3.450221; base_ce: 0.8594794; base_accuracy: 0.7202; adversarial_accuracy: 0.2226; since_best_base_ce: 1; min_base_ce: 0.8165515; 
    FastEstimator-Train: step: 7500; base_ce: 0.55798185; steps/sec: 60.22; 
    FastEstimator-Train: step: 8000; base_ce: 0.71712387; steps/sec: 61.41; 
    FastEstimator-Train: step: 8000; epoch: 8; epoch_time: 16.45 sec; 
    FastEstimator-Eval: step: 8000; epoch: 8; adv_ce: 3.7345269; base_ce: 0.8529271; base_accuracy: 0.7304; adversarial_accuracy: 0.223; since_best_base_ce: 2; min_base_ce: 0.8165515; 
    FastEstimator-Train: step: 8500; base_ce: 0.52675784; steps/sec: 60.57; 
    FastEstimator-Train: step: 9000; base_ce: 0.4638951; steps/sec: 60.67; 
    FastEstimator-Train: step: 9000; epoch: 9; epoch_time: 16.49 sec; 
    FastEstimator-Eval: step: 9000; epoch: 9; adv_ce: 3.92901; base_ce: 0.862138; base_accuracy: 0.7234; adversarial_accuracy: 0.2072; since_best_base_ce: 3; min_base_ce: 0.8165515; 
    FastEstimator-Train: step: 9500; base_ce: 0.3654891; steps/sec: 59.84; 
    FastEstimator-Train: step: 10000; base_ce: 0.24989706; steps/sec: 61.97; 
    FastEstimator-Train: step: 10000; epoch: 10; epoch_time: 16.43 sec; 
    FastEstimator-Eval: step: 10000; epoch: 10; adv_ce: 4.546957; base_ce: 0.96095663; base_accuracy: 0.7118; adversarial_accuracy: 0.1848; since_best_base_ce: 4; min_base_ce: 0.8165515; 
    FastEstimator-Train: step: 10500; base_ce: 0.3248038; steps/sec: 58.92; 
    FastEstimator-Train: step: 11000; base_ce: 0.41753396; steps/sec: 62.94; 
    FastEstimator-Train: step: 11000; epoch: 11; epoch_time: 16.43 sec; 
    FastEstimator-Eval: step: 11000; epoch: 11; adv_ce: 5.142259; base_ce: 0.99953157; base_accuracy: 0.7168; adversarial_accuracy: 0.1872; since_best_base_ce: 5; min_base_ce: 0.8165515; 
    FastEstimator-Train: step: 11500; base_ce: 0.41104344; steps/sec: 57.45; 
    FastEstimator-Train: step: 12000; base_ce: 0.26091018; steps/sec: 64.64; 
    FastEstimator-Train: step: 12000; epoch: 12; epoch_time: 16.44 sec; 
    FastEstimator-Eval: step: 12000; epoch: 12; adv_ce: 5.3693953; base_ce: 1.0091112; base_accuracy: 0.7152; adversarial_accuracy: 0.1794; since_best_base_ce: 6; min_base_ce: 0.8165515; 
    FastEstimator-Train: step: 12500; base_ce: 0.24132206; steps/sec: 60.3; 
    FastEstimator-Train: step: 13000; base_ce: 0.23491889; steps/sec: 60.01; 
    FastEstimator-Train: step: 13000; epoch: 13; epoch_time: 16.62 sec; 
    FastEstimator-Eval: step: 13000; epoch: 13; adv_ce: 6.1618037; base_ce: 1.1255364; base_accuracy: 0.7186; adversarial_accuracy: 0.179; since_best_base_ce: 7; min_base_ce: 0.8165515; 
    FastEstimator-Train: step: 13500; base_ce: 0.20184961; steps/sec: 64.37; 
    FastEstimator-Train: step: 14000; base_ce: 0.29378676; steps/sec: 73.5; 
    FastEstimator-Train: step: 14000; epoch: 14; epoch_time: 14.57 sec; 
    FastEstimator-Eval: step: 14000; epoch: 14; adv_ce: 6.383223; base_ce: 1.1051223; base_accuracy: 0.719; adversarial_accuracy: 0.174; since_best_base_ce: 8; min_base_ce: 0.8165515; 
    FastEstimator-Train: step: 14500; base_ce: 0.21271034; steps/sec: 78.73; 
    FastEstimator-Train: step: 15000; base_ce: 0.200208; steps/sec: 82.96; 
    FastEstimator-Train: step: 15000; epoch: 15; epoch_time: 12.38 sec; 
    FastEstimator-Eval: step: 15000; epoch: 15; adv_ce: 6.888222; base_ce: 1.1569692; base_accuracy: 0.7176; adversarial_accuracy: 0.1572; since_best_base_ce: 9; min_base_ce: 0.8165515; 
    FastEstimator-Train: step: 15500; base_ce: 0.2577537; steps/sec: 82.62; 
    FastEstimator-Train: step: 16000; base_ce: 0.12891757; steps/sec: 84.56; 
    FastEstimator-Train: step: 16000; epoch: 16; epoch_time: 11.97 sec; 
    FastEstimator-EarlyStopping: 'base_ce' triggered an early stop. Its best value was 0.8165515065193176 at epoch 6
    FastEstimator-Eval: step: 16000; epoch: 16; adv_ce: 7.65746; base_ce: 1.2928303; base_accuracy: 0.7038; adversarial_accuracy: 0.1596; since_best_base_ce: 10; min_base_ce: 0.8165515; 
    FastEstimator-BestModelSaver: Restoring model from /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/softmax_best_base_ce.h5
    FastEstimator-Finish: step: 16000; total_time: 277.1 sec; softmax_lr: 0.001; 
    FastEstimator-Test: step: 16000; epoch: 16; base_accuracy: 0.7188; adversarial_accuracy: 0.2392; 



```python
ecc_estimator.fit('ECC')
ecc_results = ecc_estimator.test()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 500; 
    FastEstimator-Train: step: 1; base_hinge: 0.9720483; 
    FastEstimator-Train: step: 500; base_hinge: 0.65065217; steps/sec: 86.2; 
    FastEstimator-Train: step: 1000; base_hinge: 0.48177832; steps/sec: 83.56; 
    FastEstimator-Train: step: 1000; epoch: 1; epoch_time: 12.04 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 1000; epoch: 1; adv_hinge: 0.724342; base_hinge: 0.5987196; base_accuracy: 0.4514; adversarial_accuracy: 0.2944; since_best_base_hinge: 0; min_base_hinge: 0.5987196; 
    FastEstimator-Train: step: 1500; base_hinge: 0.5693387; steps/sec: 81.56; 
    FastEstimator-Train: step: 2000; base_hinge: 0.541312; steps/sec: 80.84; 
    FastEstimator-Train: step: 2000; epoch: 2; epoch_time: 12.32 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 2000; epoch: 2; adv_hinge: 0.7137549; base_hinge: 0.54428804; base_accuracy: 0.537; adversarial_accuracy: 0.3044; since_best_base_hinge: 0; min_base_hinge: 0.54428804; 
    FastEstimator-Train: step: 2500; base_hinge: 0.55899805; steps/sec: 78.96; 
    FastEstimator-Train: step: 3000; base_hinge: 0.45505947; steps/sec: 77.87; 
    FastEstimator-Train: step: 3000; epoch: 3; epoch_time: 12.76 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 3000; epoch: 3; adv_hinge: 0.6705899; base_hinge: 0.45147446; base_accuracy: 0.6176; adversarial_accuracy: 0.3572; since_best_base_hinge: 0; min_base_hinge: 0.45147446; 
    FastEstimator-Train: step: 3500; base_hinge: 0.43476373; steps/sec: 74.89; 
    FastEstimator-Train: step: 4000; base_hinge: 0.38228527; steps/sec: 75.76; 
    FastEstimator-Train: step: 4000; epoch: 4; epoch_time: 13.27 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 4000; epoch: 4; adv_hinge: 0.68376756; base_hinge: 0.41109648; base_accuracy: 0.6318; adversarial_accuracy: 0.3414; since_best_base_hinge: 0; min_base_hinge: 0.41109648; 
    FastEstimator-Train: step: 4500; base_hinge: 0.44442576; steps/sec: 79.9; 
    FastEstimator-Train: step: 5000; base_hinge: 0.3891029; steps/sec: 82.32; 
    FastEstimator-Train: step: 5000; epoch: 5; epoch_time: 12.34 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 5000; epoch: 5; adv_hinge: 0.6567371; base_hinge: 0.36597657; base_accuracy: 0.6568; adversarial_accuracy: 0.363; since_best_base_hinge: 0; min_base_hinge: 0.36597657; 
    FastEstimator-Train: step: 5500; base_hinge: 0.3704931; steps/sec: 77.45; 
    FastEstimator-Train: step: 6000; base_hinge: 0.40984428; steps/sec: 73.35; 
    FastEstimator-Train: step: 6000; epoch: 6; epoch_time: 13.27 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 6000; epoch: 6; adv_hinge: 0.65801245; base_hinge: 0.34470206; base_accuracy: 0.678; adversarial_accuracy: 0.3656; since_best_base_hinge: 0; min_base_hinge: 0.34470206; 
    FastEstimator-Train: step: 6500; base_hinge: 0.3440132; steps/sec: 80.15; 
    FastEstimator-Train: step: 7000; base_hinge: 0.2628339; steps/sec: 83.06; 
    FastEstimator-Train: step: 7000; epoch: 7; epoch_time: 12.25 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 7000; epoch: 7; adv_hinge: 0.65256685; base_hinge: 0.32853946; base_accuracy: 0.6916; adversarial_accuracy: 0.371; since_best_base_hinge: 0; min_base_hinge: 0.32853946; 
    FastEstimator-Train: step: 7500; base_hinge: 0.18051398; steps/sec: 81.6; 
    FastEstimator-Train: step: 8000; base_hinge: 0.23352867; steps/sec: 81.08; 
    FastEstimator-Train: step: 8000; epoch: 8; epoch_time: 12.3 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 8000; epoch: 8; adv_hinge: 0.64836806; base_hinge: 0.32261345; base_accuracy: 0.6926; adversarial_accuracy: 0.37; since_best_base_hinge: 0; min_base_hinge: 0.32261345; 
    FastEstimator-Train: step: 8500; base_hinge: 0.24980038; steps/sec: 75.69; 
    FastEstimator-Train: step: 9000; base_hinge: 0.2729612; steps/sec: 74.84; 
    FastEstimator-Train: step: 9000; epoch: 9; epoch_time: 13.29 sec; 
    FastEstimator-Eval: step: 9000; epoch: 9; adv_hinge: 0.6530391; base_hinge: 0.32772756; base_accuracy: 0.6906; adversarial_accuracy: 0.3626; since_best_base_hinge: 1; min_base_hinge: 0.32261345; 
    FastEstimator-Train: step: 9500; base_hinge: 0.28272608; steps/sec: 79.62; 
    FastEstimator-Train: step: 10000; base_hinge: 0.11323662; steps/sec: 82.91; 
    FastEstimator-Train: step: 10000; epoch: 10; epoch_time: 12.3 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 10000; epoch: 10; adv_hinge: 0.6115315; base_hinge: 0.3017887; base_accuracy: 0.7164; adversarial_accuracy: 0.4026; since_best_base_hinge: 0; min_base_hinge: 0.3017887; 
    FastEstimator-Train: step: 10500; base_hinge: 0.1635123; steps/sec: 80.01; 
    FastEstimator-Train: step: 11000; base_hinge: 0.28864193; steps/sec: 82.4; 
    FastEstimator-Train: step: 11000; epoch: 11; epoch_time: 12.32 sec; 
    FastEstimator-Eval: step: 11000; epoch: 11; adv_hinge: 0.6173514; base_hinge: 0.3180277; base_accuracy: 0.6968; adversarial_accuracy: 0.3988; since_best_base_hinge: 1; min_base_hinge: 0.3017887; 
    FastEstimator-Train: step: 11500; base_hinge: 0.28502467; steps/sec: 79.03; 
    FastEstimator-Train: step: 12000; base_hinge: 0.24555106; steps/sec: 79.15; 
    FastEstimator-Train: step: 12000; epoch: 12; epoch_time: 12.65 sec; 
    FastEstimator-Eval: step: 12000; epoch: 12; adv_hinge: 0.606781; base_hinge: 0.3200605; base_accuracy: 0.6948; adversarial_accuracy: 0.4114; since_best_base_hinge: 2; min_base_hinge: 0.3017887; 
    FastEstimator-Train: step: 12500; base_hinge: 0.15662248; steps/sec: 76.64; 
    FastEstimator-Train: step: 13000; base_hinge: 0.2020681; steps/sec: 80.51; 
    FastEstimator-Train: step: 13000; epoch: 13; epoch_time: 12.73 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 13000; epoch: 13; adv_hinge: 0.5785308; base_hinge: 0.29301724; base_accuracy: 0.7168; adversarial_accuracy: 0.4378; since_best_base_hinge: 0; min_base_hinge: 0.29301724; 
    FastEstimator-Train: step: 13500; base_hinge: 0.22861297; steps/sec: 77.51; 
    FastEstimator-Train: step: 14000; base_hinge: 0.3236181; steps/sec: 81.78; 
    FastEstimator-Train: step: 14000; epoch: 14; epoch_time: 12.57 sec; 
    FastEstimator-Eval: step: 14000; epoch: 14; adv_hinge: 0.5728293; base_hinge: 0.31178313; base_accuracy: 0.7008; adversarial_accuracy: 0.4438; since_best_base_hinge: 1; min_base_hinge: 0.29301724; 
    FastEstimator-Train: step: 14500; base_hinge: 0.2981937; steps/sec: 79.19; 
    FastEstimator-Train: step: 15000; base_hinge: 0.22167464; steps/sec: 81.08; 
    FastEstimator-Train: step: 15000; epoch: 15; epoch_time: 12.48 sec; 
    FastEstimator-Eval: step: 15000; epoch: 15; adv_hinge: 0.56800824; base_hinge: 0.3034843; base_accuracy: 0.711; adversarial_accuracy: 0.453; since_best_base_hinge: 2; min_base_hinge: 0.29301724; 
    FastEstimator-Train: step: 15500; base_hinge: 0.13327494; steps/sec: 78.79; 
    FastEstimator-Train: step: 16000; base_hinge: 0.13513847; steps/sec: 83.04; 
    FastEstimator-Train: step: 16000; epoch: 16; epoch_time: 12.37 sec; 
    FastEstimator-Eval: step: 16000; epoch: 16; adv_hinge: 0.53621924; base_hinge: 0.2944945; base_accuracy: 0.7176; adversarial_accuracy: 0.4784; since_best_base_hinge: 3; min_base_hinge: 0.29301724; 
    FastEstimator-Train: step: 16500; base_hinge: 0.24307278; steps/sec: 79.35; 
    FastEstimator-Train: step: 17000; base_hinge: 0.21438818; steps/sec: 80.14; 
    FastEstimator-Train: step: 17000; epoch: 17; epoch_time: 12.55 sec; 
    FastEstimator-Eval: step: 17000; epoch: 17; adv_hinge: 0.53243375; base_hinge: 0.2946877; base_accuracy: 0.7208; adversarial_accuracy: 0.4866; since_best_base_hinge: 4; min_base_hinge: 0.29301724; 
    FastEstimator-Train: step: 17500; base_hinge: 0.2600304; steps/sec: 76.64; 
    FastEstimator-Train: step: 18000; base_hinge: 0.2710699; steps/sec: 80.69; 
    FastEstimator-Train: step: 18000; epoch: 18; epoch_time: 12.71 sec; 
    FastEstimator-Eval: step: 18000; epoch: 18; adv_hinge: 0.52874297; base_hinge: 0.3025954; base_accuracy: 0.7128; adversarial_accuracy: 0.493; since_best_base_hinge: 5; min_base_hinge: 0.29301724; 
    FastEstimator-Train: step: 18500; base_hinge: 0.1871714; steps/sec: 77.07; 
    FastEstimator-Train: step: 19000; base_hinge: 0.24277395; steps/sec: 79.41; 
    FastEstimator-Train: step: 19000; epoch: 19; epoch_time: 12.78 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 19000; epoch: 19; adv_hinge: 0.51166725; base_hinge: 0.29020032; base_accuracy: 0.7236; adversarial_accuracy: 0.5082; since_best_base_hinge: 0; min_base_hinge: 0.29020032; 
    FastEstimator-Train: step: 19500; base_hinge: 0.13240014; steps/sec: 77.01; 
    FastEstimator-Train: step: 20000; base_hinge: 0.2651819; steps/sec: 81.51; 
    FastEstimator-Train: step: 20000; epoch: 20; epoch_time: 12.63 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 20000; epoch: 20; adv_hinge: 0.49882263; base_hinge: 0.28652295; base_accuracy: 0.7258; adversarial_accuracy: 0.5216; since_best_base_hinge: 0; min_base_hinge: 0.28652295; 
    FastEstimator-Train: step: 20500; base_hinge: 0.14134462; steps/sec: 77.67; 
    FastEstimator-Train: step: 21000; base_hinge: 0.21163704; steps/sec: 81.18; 
    FastEstimator-Train: step: 21000; epoch: 21; epoch_time: 12.6 sec; 
    FastEstimator-Eval: step: 21000; epoch: 21; adv_hinge: 0.50655615; base_hinge: 0.29529086; base_accuracy: 0.7192; adversarial_accuracy: 0.513; since_best_base_hinge: 1; min_base_hinge: 0.28652295; 
    FastEstimator-Train: step: 21500; base_hinge: 0.21828929; steps/sec: 78.85; 
    FastEstimator-Train: step: 22000; base_hinge: 0.32995892; steps/sec: 83.66; 
    FastEstimator-Train: step: 22000; epoch: 22; epoch_time: 12.32 sec; 
    FastEstimator-Eval: step: 22000; epoch: 22; adv_hinge: 0.5129851; base_hinge: 0.2939868; base_accuracy: 0.7178; adversarial_accuracy: 0.5082; since_best_base_hinge: 2; min_base_hinge: 0.28652295; 
    FastEstimator-Train: step: 22500; base_hinge: 0.21359493; steps/sec: 79.11; 
    FastEstimator-Train: step: 23000; base_hinge: 0.21484666; steps/sec: 80.08; 
    FastEstimator-Train: step: 23000; epoch: 23; epoch_time: 12.56 sec; 
    FastEstimator-Eval: step: 23000; epoch: 23; adv_hinge: 0.50230604; base_hinge: 0.2944177; base_accuracy: 0.721; adversarial_accuracy: 0.52; since_best_base_hinge: 3; min_base_hinge: 0.28652295; 
    FastEstimator-Train: step: 23500; base_hinge: 0.2986081; steps/sec: 76.6; 
    FastEstimator-Train: step: 24000; base_hinge: 0.11756951; steps/sec: 81.32; 
    FastEstimator-Train: step: 24000; epoch: 24; epoch_time: 12.67 sec; 
    FastEstimator-Eval: step: 24000; epoch: 24; adv_hinge: 0.49573535; base_hinge: 0.29334903; base_accuracy: 0.718; adversarial_accuracy: 0.526; since_best_base_hinge: 4; min_base_hinge: 0.28652295; 
    FastEstimator-Train: step: 24500; base_hinge: 0.24381065; steps/sec: 76.59; 
    FastEstimator-Train: step: 25000; base_hinge: 0.1649466; steps/sec: 79.48; 
    FastEstimator-Train: step: 25000; epoch: 25; epoch_time: 12.82 sec; 
    FastEstimator-Eval: step: 25000; epoch: 25; adv_hinge: 0.4862012; base_hinge: 0.30284664; base_accuracy: 0.7108; adversarial_accuracy: 0.5342; since_best_base_hinge: 5; min_base_hinge: 0.28652295; 
    FastEstimator-Train: step: 25500; base_hinge: 0.16841447; steps/sec: 77.41; 
    FastEstimator-Train: step: 26000; base_hinge: 0.16662271; steps/sec: 81.66; 
    FastEstimator-Train: step: 26000; epoch: 26; epoch_time: 12.58 sec; 
    FastEstimator-Eval: step: 26000; epoch: 26; adv_hinge: 0.47293594; base_hinge: 0.29134712; base_accuracy: 0.7232; adversarial_accuracy: 0.548; since_best_base_hinge: 6; min_base_hinge: 0.28652295; 
    FastEstimator-Train: step: 26500; base_hinge: 0.21119633; steps/sec: 77.31; 
    FastEstimator-Train: step: 27000; base_hinge: 0.17389616; steps/sec: 81.15; 
    FastEstimator-Train: step: 27000; epoch: 27; epoch_time: 12.63 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 27000; epoch: 27; adv_hinge: 0.46093524; base_hinge: 0.28287578; base_accuracy: 0.7288; adversarial_accuracy: 0.5614; since_best_base_hinge: 0; min_base_hinge: 0.28287578; 
    FastEstimator-Train: step: 27500; base_hinge: 0.05262907; steps/sec: 78.62; 
    FastEstimator-Train: step: 28000; base_hinge: 0.14051457; steps/sec: 82.26; 
    FastEstimator-Train: step: 28000; epoch: 28; epoch_time: 12.44 sec; 
    FastEstimator-Eval: step: 28000; epoch: 28; adv_hinge: 0.4624555; base_hinge: 0.28717962; base_accuracy: 0.7276; adversarial_accuracy: 0.5606; since_best_base_hinge: 1; min_base_hinge: 0.28287578; 
    FastEstimator-Train: step: 28500; base_hinge: 0.06206832; steps/sec: 80.6; 
    FastEstimator-Train: step: 29000; base_hinge: 0.22145921; steps/sec: 82.87; 
    FastEstimator-Train: step: 29000; epoch: 29; epoch_time: 12.24 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 29000; epoch: 29; adv_hinge: 0.456716; base_hinge: 0.28246412; base_accuracy: 0.7322; adversarial_accuracy: 0.564; since_best_base_hinge: 0; min_base_hinge: 0.28246412; 
    FastEstimator-Train: step: 29500; base_hinge: 0.15014073; steps/sec: 77.42; 
    FastEstimator-Train: step: 30000; base_hinge: 0.20060506; steps/sec: 80.7; 
    FastEstimator-Train: step: 30000; epoch: 30; epoch_time: 12.65 sec; 
    FastEstimator-Eval: step: 30000; epoch: 30; adv_hinge: 0.4513559; base_hinge: 0.2847252; base_accuracy: 0.7274; adversarial_accuracy: 0.57; since_best_base_hinge: 1; min_base_hinge: 0.28246412; 
    FastEstimator-Train: step: 30500; base_hinge: 0.15612331; steps/sec: 76.59; 
    FastEstimator-Train: step: 31000; base_hinge: 0.2618798; steps/sec: 81.39; 
    FastEstimator-Train: step: 31000; epoch: 31; epoch_time: 12.67 sec; 
    FastEstimator-Eval: step: 31000; epoch: 31; adv_hinge: 0.44176352; base_hinge: 0.28683442; base_accuracy: 0.7302; adversarial_accuracy: 0.5806; since_best_base_hinge: 2; min_base_hinge: 0.28246412; 
    FastEstimator-Train: step: 31500; base_hinge: 0.13087274; steps/sec: 76.98; 
    FastEstimator-Train: step: 32000; base_hinge: 0.22726895; steps/sec: 80.06; 
    FastEstimator-Train: step: 32000; epoch: 32; epoch_time: 12.74 sec; 
    FastEstimator-Eval: step: 32000; epoch: 32; adv_hinge: 0.4663296; base_hinge: 0.29836673; base_accuracy: 0.7204; adversarial_accuracy: 0.557; since_best_base_hinge: 3; min_base_hinge: 0.28246412; 
    FastEstimator-Train: step: 32500; base_hinge: 0.17779167; steps/sec: 77.87; 
    FastEstimator-Train: step: 33000; base_hinge: 0.14685188; steps/sec: 80.8; 
    FastEstimator-Train: step: 33000; epoch: 33; epoch_time: 12.61 sec; 
    FastEstimator-Eval: step: 33000; epoch: 33; adv_hinge: 0.44809714; base_hinge: 0.28558087; base_accuracy: 0.7282; adversarial_accuracy: 0.5716; since_best_base_hinge: 4; min_base_hinge: 0.28246412; 
    FastEstimator-Train: step: 33500; base_hinge: 0.18333197; steps/sec: 77.47; 
    FastEstimator-Train: step: 34000; base_hinge: 0.1574696; steps/sec: 80.22; 
    FastEstimator-Train: step: 34000; epoch: 34; epoch_time: 12.69 sec; 
    FastEstimator-Eval: step: 34000; epoch: 34; adv_hinge: 0.44056252; base_hinge: 0.28778878; base_accuracy: 0.7264; adversarial_accuracy: 0.5796; since_best_base_hinge: 5; min_base_hinge: 0.28246412; 
    FastEstimator-Train: step: 34500; base_hinge: 0.21325463; steps/sec: 77.05; 
    FastEstimator-Train: step: 35000; base_hinge: 0.20648474; steps/sec: 80.68; 
    FastEstimator-Train: step: 35000; epoch: 35; epoch_time: 12.69 sec; 
    FastEstimator-Eval: step: 35000; epoch: 35; adv_hinge: 0.42195377; base_hinge: 0.2834217; base_accuracy: 0.7322; adversarial_accuracy: 0.6006; since_best_base_hinge: 6; min_base_hinge: 0.28246412; 
    FastEstimator-Train: step: 35500; base_hinge: 0.14986329; steps/sec: 76.76; 
    FastEstimator-Train: step: 36000; base_hinge: 0.1910067; steps/sec: 80.55; 
    FastEstimator-Train: step: 36000; epoch: 36; epoch_time: 12.72 sec; 
    FastEstimator-Eval: step: 36000; epoch: 36; adv_hinge: 0.43846917; base_hinge: 0.28581482; base_accuracy: 0.7264; adversarial_accuracy: 0.5842; since_best_base_hinge: 7; min_base_hinge: 0.28246412; 
    FastEstimator-Train: step: 36500; base_hinge: 0.093400106; steps/sec: 76.1; 
    FastEstimator-Train: step: 37000; base_hinge: 0.20568691; steps/sec: 79.59; 
    FastEstimator-Train: step: 37000; epoch: 37; epoch_time: 12.85 sec; 
    FastEstimator-Eval: step: 37000; epoch: 37; adv_hinge: 0.42621788; base_hinge: 0.28850666; base_accuracy: 0.7252; adversarial_accuracy: 0.5944; since_best_base_hinge: 8; min_base_hinge: 0.28246412; 
    FastEstimator-Train: step: 37500; base_hinge: 0.14697212; steps/sec: 76.41; 
    FastEstimator-Train: step: 38000; base_hinge: 0.17251854; steps/sec: 79.85; 
    FastEstimator-Train: step: 38000; epoch: 38; epoch_time: 12.81 sec; 
    FastEstimator-Eval: step: 38000; epoch: 38; adv_hinge: 0.4249133; base_hinge: 0.28864872; base_accuracy: 0.7222; adversarial_accuracy: 0.5958; since_best_base_hinge: 9; min_base_hinge: 0.28246412; 
    FastEstimator-Train: step: 38500; base_hinge: 0.1266299; steps/sec: 78.27; 
    FastEstimator-Train: step: 39000; base_hinge: 0.14579202; steps/sec: 81.51; 
    FastEstimator-Train: step: 39000; epoch: 39; epoch_time: 12.52 sec; 
    FastEstimator-EarlyStopping: 'base_hinge' triggered an early stop. Its best value was 0.2824641168117523 at epoch 29
    FastEstimator-Eval: step: 39000; epoch: 39; adv_hinge: 0.4217629; base_hinge: 0.28681317; base_accuracy: 0.7264; adversarial_accuracy: 0.5978; since_best_base_hinge: 10; min_base_hinge: 0.28246412; 
    FastEstimator-BestModelSaver: Restoring model from /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/ecc_best_base_hinge.h5
    FastEstimator-Finish: step: 39000; total_time: 562.65 sec; ecc_lr: 0.001; 
    FastEstimator-Test: step: 39000; epoch: 39; base_accuracy: 0.724; adversarial_accuracy: 0.5696; 



```python
hydra_estimator.fit('Hydra')
hydra_results = hydra_estimator.test()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 500; 
    FastEstimator-Train: step: 1; base_hinge: 1.007727; 
    FastEstimator-Train: step: 500; base_hinge: 0.7029223; steps/sec: 79.37; 
    FastEstimator-Train: step: 1000; base_hinge: 0.5938768; steps/sec: 80.64; 
    FastEstimator-Train: step: 1000; epoch: 1; epoch_time: 12.92 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 1000; epoch: 1; adv_hinge: 0.7107463; base_hinge: 0.58592; base_accuracy: 0.444; adversarial_accuracy: 0.2916; since_best_base_hinge: 0; min_base_hinge: 0.58592; 
    FastEstimator-Train: step: 1500; base_hinge: 0.5222081; steps/sec: 77.67; 
    FastEstimator-Train: step: 2000; base_hinge: 0.49780965; steps/sec: 78.4; 
    FastEstimator-Train: step: 2000; epoch: 2; epoch_time: 12.82 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 2000; epoch: 2; adv_hinge: 0.6803317; base_hinge: 0.5114545; base_accuracy: 0.5206; adversarial_accuracy: 0.3412; since_best_base_hinge: 0; min_base_hinge: 0.5114545; 
    FastEstimator-Train: step: 2500; base_hinge: 0.5207081; steps/sec: 78.19; 
    FastEstimator-Train: step: 3000; base_hinge: 0.46403983; steps/sec: 79.88; 
    FastEstimator-Train: step: 3000; epoch: 3; epoch_time: 12.65 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 3000; epoch: 3; adv_hinge: 0.6644043; base_hinge: 0.46950454; base_accuracy: 0.567; adversarial_accuracy: 0.3488; since_best_base_hinge: 0; min_base_hinge: 0.46950454; 
    FastEstimator-Train: step: 3500; base_hinge: 0.40603283; steps/sec: 75.42; 
    FastEstimator-Train: step: 4000; base_hinge: 0.3923445; steps/sec: 77.49; 
    FastEstimator-Train: step: 4000; epoch: 4; epoch_time: 13.09 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 4000; epoch: 4; adv_hinge: 0.64425445; base_hinge: 0.43538117; base_accuracy: 0.5906; adversarial_accuracy: 0.381; since_best_base_hinge: 0; min_base_hinge: 0.43538117; 
    FastEstimator-Train: step: 4500; base_hinge: 0.39230484; steps/sec: 78.23; 
    FastEstimator-Train: step: 5000; base_hinge: 0.3513677; steps/sec: 80.93; 
    FastEstimator-Train: step: 5000; epoch: 5; epoch_time: 12.57 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 5000; epoch: 5; adv_hinge: 0.63775545; base_hinge: 0.41264045; base_accuracy: 0.622; adversarial_accuracy: 0.3874; since_best_base_hinge: 0; min_base_hinge: 0.41264045; 
    FastEstimator-Train: step: 5500; base_hinge: 0.333709; steps/sec: 77.66; 
    FastEstimator-Train: step: 6000; base_hinge: 0.42386734; steps/sec: 79.09; 
    FastEstimator-Train: step: 6000; epoch: 6; epoch_time: 12.75 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 6000; epoch: 6; adv_hinge: 0.63049006; base_hinge: 0.38785324; base_accuracy: 0.651; adversarial_accuracy: 0.39; since_best_base_hinge: 0; min_base_hinge: 0.38785324; 
    FastEstimator-Train: step: 6500; base_hinge: 0.3881634; steps/sec: 78.18; 
    FastEstimator-Train: step: 7000; base_hinge: 0.4034718; steps/sec: 79.46; 
    FastEstimator-Train: step: 7000; epoch: 7; epoch_time: 12.69 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 7000; epoch: 7; adv_hinge: 0.641855; base_hinge: 0.38448378; base_accuracy: 0.647; adversarial_accuracy: 0.381; since_best_base_hinge: 0; min_base_hinge: 0.38448378; 
    FastEstimator-Train: step: 7500; base_hinge: 0.34052792; steps/sec: 80.53; 
    FastEstimator-Train: step: 8000; base_hinge: 0.43806535; steps/sec: 83.03; 
    FastEstimator-Train: step: 8000; epoch: 8; epoch_time: 12.23 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 8000; epoch: 8; adv_hinge: 0.62571025; base_hinge: 0.36942664; base_accuracy: 0.6638; adversarial_accuracy: 0.3916; since_best_base_hinge: 0; min_base_hinge: 0.36942664; 
    FastEstimator-Train: step: 8500; base_hinge: 0.33974144; steps/sec: 78.38; 
    FastEstimator-Train: step: 9000; base_hinge: 0.31150648; steps/sec: 71.43; 
    FastEstimator-Train: step: 9000; epoch: 9; epoch_time: 13.38 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 9000; epoch: 9; adv_hinge: 0.6160934; base_hinge: 0.3464116; base_accuracy: 0.6832; adversarial_accuracy: 0.4014; since_best_base_hinge: 0; min_base_hinge: 0.3464116; 
    FastEstimator-Train: step: 9500; base_hinge: 0.23986004; steps/sec: 77.28; 
    FastEstimator-Train: step: 10000; base_hinge: 0.41196102; steps/sec: 81.07; 
    FastEstimator-Train: step: 10000; epoch: 10; epoch_time: 12.66 sec; 
    FastEstimator-Eval: step: 10000; epoch: 10; adv_hinge: 0.6257987; base_hinge: 0.35541627; base_accuracy: 0.6738; adversarial_accuracy: 0.3974; since_best_base_hinge: 1; min_base_hinge: 0.3464116; 
    FastEstimator-Train: step: 10500; base_hinge: 0.26076213; steps/sec: 75.51; 
    FastEstimator-Train: step: 11000; base_hinge: 0.32601464; steps/sec: 75.36; 
    FastEstimator-Train: step: 11000; epoch: 11; epoch_time: 13.23 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 11000; epoch: 11; adv_hinge: 0.62028766; base_hinge: 0.33794487; base_accuracy: 0.6956; adversarial_accuracy: 0.3998; since_best_base_hinge: 0; min_base_hinge: 0.33794487; 
    FastEstimator-Train: step: 11500; base_hinge: 0.21475668; steps/sec: 72.14; 
    FastEstimator-Train: step: 12000; base_hinge: 0.3604712; steps/sec: 70.87; 
    FastEstimator-Train: step: 12000; epoch: 12; epoch_time: 13.99 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 12000; epoch: 12; adv_hinge: 0.6220209; base_hinge: 0.33451056; base_accuracy: 0.697; adversarial_accuracy: 0.3944; since_best_base_hinge: 0; min_base_hinge: 0.33451056; 
    FastEstimator-Train: step: 12500; base_hinge: 0.30068347; steps/sec: 73.65; 
    FastEstimator-Train: step: 13000; base_hinge: 0.30370423; steps/sec: 75.43; 
    FastEstimator-Train: step: 13000; epoch: 13; epoch_time: 13.42 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 13000; epoch: 13; adv_hinge: 0.616508; base_hinge: 0.33238065; base_accuracy: 0.6948; adversarial_accuracy: 0.3986; since_best_base_hinge: 0; min_base_hinge: 0.33238065; 
    FastEstimator-Train: step: 13500; base_hinge: 0.30079776; steps/sec: 74.79; 
    FastEstimator-Train: step: 14000; base_hinge: 0.2283918; steps/sec: 78.21; 
    FastEstimator-Train: step: 14000; epoch: 14; epoch_time: 13.07 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 14000; epoch: 14; adv_hinge: 0.6049673; base_hinge: 0.33148763; base_accuracy: 0.698; adversarial_accuracy: 0.409; since_best_base_hinge: 0; min_base_hinge: 0.33148763; 
    FastEstimator-Train: step: 14500; base_hinge: 0.3066492; steps/sec: 74.76; 
    FastEstimator-Train: step: 15000; base_hinge: 0.2918469; steps/sec: 77.99; 
    FastEstimator-Train: step: 15000; epoch: 15; epoch_time: 13.1 sec; 
    FastEstimator-Eval: step: 15000; epoch: 15; adv_hinge: 0.6270892; base_hinge: 0.33353606; base_accuracy: 0.7; adversarial_accuracy: 0.386; since_best_base_hinge: 1; min_base_hinge: 0.33148763; 
    FastEstimator-Train: step: 15500; base_hinge: 0.24441415; steps/sec: 75.55; 
    FastEstimator-Train: step: 16000; base_hinge: 0.22781277; steps/sec: 78.93; 
    FastEstimator-Train: step: 16000; epoch: 16; epoch_time: 12.95 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 16000; epoch: 16; adv_hinge: 0.6047035; base_hinge: 0.32660407; base_accuracy: 0.7004; adversarial_accuracy: 0.4098; since_best_base_hinge: 0; min_base_hinge: 0.32660407; 
    FastEstimator-Train: step: 16500; base_hinge: 0.2558497; steps/sec: 75.78; 
    FastEstimator-Train: step: 17000; base_hinge: 0.2882976; steps/sec: 79.81; 
    FastEstimator-Train: step: 17000; epoch: 17; epoch_time: 12.86 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 17000; epoch: 17; adv_hinge: 0.6056667; base_hinge: 0.3208551; base_accuracy: 0.7022; adversarial_accuracy: 0.4028; since_best_base_hinge: 0; min_base_hinge: 0.3208551; 
    FastEstimator-Train: step: 17500; base_hinge: 0.20751992; steps/sec: 76.49; 
    FastEstimator-Train: step: 18000; base_hinge: 0.16293184; steps/sec: 79.81; 
    FastEstimator-Train: step: 18000; epoch: 18; epoch_time: 12.8 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 18000; epoch: 18; adv_hinge: 0.5919955; base_hinge: 0.31459105; base_accuracy: 0.7128; adversarial_accuracy: 0.42; since_best_base_hinge: 0; min_base_hinge: 0.31459105; 
    FastEstimator-Train: step: 18500; base_hinge: 0.2297887; steps/sec: 75.47; 
    FastEstimator-Train: step: 19000; base_hinge: 0.2184592; steps/sec: 77.62; 
    FastEstimator-Train: step: 19000; epoch: 19; epoch_time: 13.07 sec; 
    FastEstimator-Eval: step: 19000; epoch: 19; adv_hinge: 0.59401447; base_hinge: 0.31972674; base_accuracy: 0.7112; adversarial_accuracy: 0.414; since_best_base_hinge: 1; min_base_hinge: 0.31459105; 
    FastEstimator-Train: step: 19500; base_hinge: 0.2566901; steps/sec: 75.37; 
    FastEstimator-Train: step: 20000; base_hinge: 0.31192818; steps/sec: 79.2; 
    FastEstimator-Train: step: 20000; epoch: 20; epoch_time: 12.95 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 20000; epoch: 20; adv_hinge: 0.58349407; base_hinge: 0.31102765; base_accuracy: 0.7166; adversarial_accuracy: 0.429; since_best_base_hinge: 0; min_base_hinge: 0.31102765; 
    FastEstimator-Train: step: 20500; base_hinge: 0.33048096; steps/sec: 75.77; 
    FastEstimator-Train: step: 21000; base_hinge: 0.22020546; steps/sec: 78.41; 
    FastEstimator-Train: step: 21000; epoch: 21; epoch_time: 12.98 sec; 
    FastEstimator-Eval: step: 21000; epoch: 21; adv_hinge: 0.57951987; base_hinge: 0.3130729; base_accuracy: 0.7154; adversarial_accuracy: 0.4266; since_best_base_hinge: 1; min_base_hinge: 0.31102765; 
    FastEstimator-Train: step: 21500; base_hinge: 0.2436388; steps/sec: 75.94; 
    FastEstimator-Train: step: 22000; base_hinge: 0.19141665; steps/sec: 78.86; 
    FastEstimator-Train: step: 22000; epoch: 22; epoch_time: 12.92 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 22000; epoch: 22; adv_hinge: 0.5714458; base_hinge: 0.31004772; base_accuracy: 0.7156; adversarial_accuracy: 0.4366; since_best_base_hinge: 0; min_base_hinge: 0.31004772; 
    FastEstimator-Train: step: 22500; base_hinge: 0.2346386; steps/sec: 75.87; 
    FastEstimator-Train: step: 23000; base_hinge: 0.2227555; steps/sec: 78.81; 
    FastEstimator-Train: step: 23000; epoch: 23; epoch_time: 12.94 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 23000; epoch: 23; adv_hinge: 0.5728474; base_hinge: 0.3098317; base_accuracy: 0.7176; adversarial_accuracy: 0.4386; since_best_base_hinge: 0; min_base_hinge: 0.3098317; 
    FastEstimator-Train: step: 23500; base_hinge: 0.163346; steps/sec: 74.76; 
    FastEstimator-Train: step: 24000; base_hinge: 0.2146762; steps/sec: 76.85; 
    FastEstimator-Train: step: 24000; epoch: 24; epoch_time: 13.19 sec; 
    FastEstimator-Eval: step: 24000; epoch: 24; adv_hinge: 0.5920725; base_hinge: 0.3194797; base_accuracy: 0.7064; adversarial_accuracy: 0.4198; since_best_base_hinge: 1; min_base_hinge: 0.3098317; 
    FastEstimator-Train: step: 24500; base_hinge: 0.26413205; steps/sec: 74.6; 
    FastEstimator-Train: step: 25000; base_hinge: 0.19393927; steps/sec: 76.53; 
    FastEstimator-Train: step: 25000; epoch: 25; epoch_time: 13.24 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 25000; epoch: 25; adv_hinge: 0.55823886; base_hinge: 0.3062974; base_accuracy: 0.7196; adversarial_accuracy: 0.4534; since_best_base_hinge: 0; min_base_hinge: 0.3062974; 
    FastEstimator-Train: step: 25500; base_hinge: 0.22403201; steps/sec: 73.96; 
    FastEstimator-Train: step: 26000; base_hinge: 0.26596522; steps/sec: 77.8; 
    FastEstimator-Train: step: 26000; epoch: 26; epoch_time: 13.19 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Eval: step: 26000; epoch: 26; adv_hinge: 0.55658454; base_hinge: 0.29793462; base_accuracy: 0.7272; adversarial_accuracy: 0.4524; since_best_base_hinge: 0; min_base_hinge: 0.29793462; 
    FastEstimator-Train: step: 26500; base_hinge: 0.17813882; steps/sec: 74.35; 
    FastEstimator-Train: step: 27000; base_hinge: 0.21485883; steps/sec: 78.4; 
    FastEstimator-Train: step: 27000; epoch: 27; epoch_time: 13.1 sec; 
    FastEstimator-Eval: step: 27000; epoch: 27; adv_hinge: 0.5723389; base_hinge: 0.31458738; base_accuracy: 0.707; adversarial_accuracy: 0.44; since_best_base_hinge: 1; min_base_hinge: 0.29793462; 
    FastEstimator-Train: step: 27500; base_hinge: 0.15185395; steps/sec: 74.27; 
    FastEstimator-Train: step: 28000; base_hinge: 0.27865165; steps/sec: 79.26; 
    FastEstimator-Train: step: 28000; epoch: 28; epoch_time: 13.04 sec; 
    FastEstimator-Eval: step: 28000; epoch: 28; adv_hinge: 0.5735497; base_hinge: 0.30879247; base_accuracy: 0.7202; adversarial_accuracy: 0.4356; since_best_base_hinge: 2; min_base_hinge: 0.29793462; 
    FastEstimator-Train: step: 28500; base_hinge: 0.24824846; steps/sec: 76.32; 
    FastEstimator-Train: step: 29000; base_hinge: 0.17269535; steps/sec: 77.05; 
    FastEstimator-Train: step: 29000; epoch: 29; epoch_time: 13.04 sec; 
    FastEstimator-Eval: step: 29000; epoch: 29; adv_hinge: 0.5488098; base_hinge: 0.30959663; base_accuracy: 0.7142; adversarial_accuracy: 0.463; since_best_base_hinge: 3; min_base_hinge: 0.29793462; 
    FastEstimator-Train: step: 29500; base_hinge: 0.136942; steps/sec: 75.83; 
    FastEstimator-Train: step: 30000; base_hinge: 0.23609023; steps/sec: 75.05; 
    FastEstimator-Train: step: 30000; epoch: 30; epoch_time: 13.26 sec; 
    FastEstimator-Eval: step: 30000; epoch: 30; adv_hinge: 0.5553243; base_hinge: 0.3113994; base_accuracy: 0.7178; adversarial_accuracy: 0.4566; since_best_base_hinge: 4; min_base_hinge: 0.29793462; 
    FastEstimator-Train: step: 30500; base_hinge: 0.1271786; steps/sec: 76.69; 
    FastEstimator-Train: step: 31000; base_hinge: 0.14635064; steps/sec: 78.81; 
    FastEstimator-Train: step: 31000; epoch: 31; epoch_time: 12.86 sec; 
    FastEstimator-Eval: step: 31000; epoch: 31; adv_hinge: 0.54818094; base_hinge: 0.3015276; base_accuracy: 0.7248; adversarial_accuracy: 0.4672; since_best_base_hinge: 5; min_base_hinge: 0.29793462; 
    FastEstimator-Train: step: 31500; base_hinge: 0.3047498; steps/sec: 74.89; 
    FastEstimator-Train: step: 32000; base_hinge: 0.17989486; steps/sec: 77.08; 
    FastEstimator-Train: step: 32000; epoch: 32; epoch_time: 13.17 sec; 
    FastEstimator-Eval: step: 32000; epoch: 32; adv_hinge: 0.5483753; base_hinge: 0.30732492; base_accuracy: 0.7166; adversarial_accuracy: 0.4668; since_best_base_hinge: 6; min_base_hinge: 0.29793462; 
    FastEstimator-Train: step: 32500; base_hinge: 0.19096829; steps/sec: 75.16; 
    FastEstimator-Train: step: 33000; base_hinge: 0.18576458; steps/sec: 77.72; 
    FastEstimator-Train: step: 33000; epoch: 33; epoch_time: 13.08 sec; 
    FastEstimator-Eval: step: 33000; epoch: 33; adv_hinge: 0.5341504; base_hinge: 0.30691355; base_accuracy: 0.7164; adversarial_accuracy: 0.4812; since_best_base_hinge: 7; min_base_hinge: 0.29793462; 
    FastEstimator-Train: step: 33500; base_hinge: 0.124624774; steps/sec: 75.69; 
    FastEstimator-Train: step: 34000; base_hinge: 0.24719924; steps/sec: 78.77; 
    FastEstimator-Train: step: 34000; epoch: 34; epoch_time: 12.96 sec; 
    FastEstimator-Eval: step: 34000; epoch: 34; adv_hinge: 0.53387624; base_hinge: 0.30544707; base_accuracy: 0.7218; adversarial_accuracy: 0.4796; since_best_base_hinge: 8; min_base_hinge: 0.29793462; 
    FastEstimator-Train: step: 34500; base_hinge: 0.2165949; steps/sec: 75.61; 
    FastEstimator-Train: step: 35000; base_hinge: 0.17847629; steps/sec: 80.94; 
    FastEstimator-Train: step: 35000; epoch: 35; epoch_time: 12.79 sec; 
    FastEstimator-Eval: step: 35000; epoch: 35; adv_hinge: 0.54723406; base_hinge: 0.30917642; base_accuracy: 0.7202; adversarial_accuracy: 0.4658; since_best_base_hinge: 9; min_base_hinge: 0.29793462; 
    FastEstimator-Train: step: 35500; base_hinge: 0.23268871; steps/sec: 75.68; 
    FastEstimator-Train: step: 36000; base_hinge: 0.17250317; steps/sec: 79.06; 
    FastEstimator-Train: step: 36000; epoch: 36; epoch_time: 12.93 sec; 
    FastEstimator-EarlyStopping: 'base_hinge' triggered an early stop. Its best value was 0.2979346215724945 at epoch 26
    FastEstimator-Eval: step: 36000; epoch: 36; adv_hinge: 0.5309303; base_hinge: 0.30545956; base_accuracy: 0.7232; adversarial_accuracy: 0.487; since_best_base_hinge: 10; min_base_hinge: 0.29793462; 
    FastEstimator-BestModelSaver: Restoring model from /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpbikpoisy/hydra_ecc_best_base_hinge.h5
    FastEstimator-Finish: step: 36000; total_time: 536.31 sec; hydra_ecc_lr: 0.001; 
    FastEstimator-Test: step: 36000; epoch: 36; base_accuracy: 0.7302; adversarial_accuracy: 0.461; 


## Comparing the Results


```python
logs = visualize_logs(experiments=[softmax_results, ecc_results, hydra_results], ignore_metrics={'ecc_lr', 'hydra_ecc_lr', 'softmax_lr', 'logging_interval', 'num_device', 'epoch_time', 'min_base_ce', 'adv_ce', 'total_time'})
```


![png](assets/branches/master/example/adversarial_training/ecc_hinge_files/ecc_hinge_23_0.png)


As you can see, the conventional network using softmax to convert logits to class probabilities actually gets more and more vulnerable to adversarial attacks as training progresses. It also quickly overfits to the data, reaching an optimal performance around epoch 7. By switching the output layer of the model to generate an error correcting code and training with hinge loss, the model is able to train almost 6 times longer before reaching peak conventional accuracy. Moreover, the adversarial performance of the network continues to improve even after the main training runs out. This is significantly better performance than networks trained specifically to combat this attack, shown in the [FGSM](../fgsm/fgsm.ipynb) notebook. It can also be seen that there is no additional cost to training using ECC as opposed to softmax in terms of steps/sec. This is a big benefit over FGSM, where the training time for each step is doubled. With these benefits in mind, you may want to consider never using softmax again.
