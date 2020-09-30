# Adversarial Robustness with Error Correcting Codes
## (Never use Softmax again)

In this example we will show how using error correcting codes to convert model logits to probabilities can drastically reduce model overfitting while simultaneously increasing model robustness against adversarial attacks. In other words, why you should never use a softmax layer again. This phenomena was first publicized by the US Army in a [2019 Neurips Paper](https://papers.nips.cc/paper/9070-error-correcting-output-codes-improve-probability-estimation-and-adversarial-robustness-of-deep-neural-networks.pdf). For background on adversarial attacks, and on the attack type we will be demonstrating here, check out our [FGSM](../fgsm/fgsm.ipynb) apphub example. Note that in this apphub we will not be training against adversarial samples, but only performing adversarial attacks during evaluation to see how different models fair against them.

## Imports


```python
import math
import tempfile

from tensorflow.python.keras import Sequential, layers
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.python.keras.models import Model

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import cifar10
from fastestimator.layers.tensorflow import HadamardCode
from fastestimator.op.numpyop.univariate import Normalize
from fastestimator.op.tensorop import Average
from fastestimator.op.tensorop.gradient import FGSM, Watch
from fastestimator.op.tensorop.loss import CrossEntropy
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
    ops=[Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))])
```

## Defining an Estimator
In this apphub we will be comparing three very similar models, all using the same training and evaluation routines. Hence a function to generate the estimators:


```python
def get_estimator(model):
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
        Accuracy(true_key="y", pred_key="y_pred", output_name="base accuracy"),
        Accuracy(true_key="y", pred_key="y_pred_adv", output_name="adversarial accuracy"),
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

## The Models
### 1 - A LeNet model with Softmax


```python
softmax_model = fe.build(model_fn=lambda:LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam", model_name='softmax')
```

### 2 - A LeNet model with Error Correcting Codes


```python
def EccLeNet(input_shape=(32, 32, 3), classes=10):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(HadamardCode(classes))  # Note that this is the only difference between this model and the FE LeNet implementation
    return model
```


```python
ecc_model = fe.build(model_fn=EccLeNet, optimizer_fn="adam", model_name='ecc')
```

### 3 - A LeNet model using ECC and multiple feature heads
While it is common practice to follow the feature extraction layers of convolution networks with several fully connected layers in order to perform classification, this can lead to the final logits being interdependent which can actually reduce the robustness of the network. One way around this is to divide your classification layers into multiple smaller independent units:


```python
def HydraEccLeNet(input_shape=(32, 32, 3), classes=10):
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
    outputs = HadamardCode(classes)(heads)
    return Model(inputs=inputs, outputs=outputs)
```


```python
hydra_model = fe.build(model_fn=HydraEccLeNet, optimizer_fn="adam", model_name='hydra_ecc')
```

## The Experiments
Let's get Estimators for each of these models and compare them:


```python
softmax_estimator = get_estimator(softmax_model)
ecc_estimator = get_estimator(ecc_model)
hydra_estimator = get_estimator(hydra_model)
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
    FastEstimator-Train: step: 1; base_ce: 2.3754928; 
    FastEstimator-Train: step: 500; base_ce: 1.2483782; steps/sec: 33.49; 
    FastEstimator-Train: step: 1000; base_ce: 1.3786774; steps/sec: 24.76; 
    FastEstimator-Train: step: 1000; epoch: 1; epoch_time: 36.88 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 1000; epoch: 1; base_ce: 1.1597806; adv_ce: 1.9426155; base accuracy: 0.5904; adversarial accuracy: 0.2798; since_best_base_ce: 0; min_base_ce: 1.1597806; 
    FastEstimator-Train: step: 1500; base_ce: 1.051444; steps/sec: 28.7; 
    FastEstimator-Train: step: 2000; base_ce: 1.1310353; steps/sec: 24.35; 
    FastEstimator-Train: step: 2000; epoch: 2; epoch_time: 38.0 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 2000; epoch: 2; base_ce: 0.9508422; adv_ce: 2.08377; base accuracy: 0.6692; adversarial accuracy: 0.2832; since_best_base_ce: 0; min_base_ce: 0.9508422; 
    FastEstimator-Train: step: 2500; base_ce: 0.80142707; steps/sec: 22.01; 
    FastEstimator-Train: step: 3000; base_ce: 0.7681661; steps/sec: 20.49; 
    FastEstimator-Train: step: 3000; epoch: 3; epoch_time: 47.08 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 3000; epoch: 3; base_ce: 0.8998893; adv_ce: 2.2634797; base accuracy: 0.691; adversarial accuracy: 0.2908; since_best_base_ce: 0; min_base_ce: 0.8998893; 
    FastEstimator-Train: step: 3500; base_ce: 0.78990793; steps/sec: 19.68; 
    FastEstimator-Train: step: 4000; base_ce: 1.1161602; steps/sec: 21.64; 
    FastEstimator-Train: step: 4000; epoch: 4; epoch_time: 48.53 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 4000; epoch: 4; base_ce: 0.83236206; adv_ce: 2.4632723; base accuracy: 0.712; adversarial accuracy: 0.2818; since_best_base_ce: 0; min_base_ce: 0.83236206; 
    FastEstimator-Train: step: 4500; base_ce: 0.80196106; steps/sec: 20.12; 
    FastEstimator-Train: step: 5000; base_ce: 0.75015104; steps/sec: 21.42; 
    FastEstimator-Train: step: 5000; epoch: 5; epoch_time: 48.18 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 5000; epoch: 5; base_ce: 0.8077117; adv_ce: 2.6284113; base accuracy: 0.7242; adversarial accuracy: 0.2666; since_best_base_ce: 0; min_base_ce: 0.8077117; 
    FastEstimator-Train: step: 5500; base_ce: 0.82036537; steps/sec: 23.3; 
    FastEstimator-Train: step: 6000; base_ce: 0.70514345; steps/sec: 25.92; 
    FastEstimator-Train: step: 6000; epoch: 6; epoch_time: 40.74 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 6000; epoch: 6; base_ce: 0.79506516; adv_ce: 2.9957755; base accuracy: 0.7246; adversarial accuracy: 0.256; since_best_base_ce: 0; min_base_ce: 0.79506516; 
    FastEstimator-Train: step: 6500; base_ce: 0.54250854; steps/sec: 26.72; 
    FastEstimator-Train: step: 7000; base_ce: 0.68669426; steps/sec: 28.35; 
    FastEstimator-Train: step: 7000; epoch: 7; epoch_time: 36.35 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/softmax_best_base_ce.h5
    FastEstimator-Eval: step: 7000; epoch: 7; base_ce: 0.78500015; adv_ce: 3.0124106; base accuracy: 0.7364; adversarial accuracy: 0.2438; since_best_base_ce: 0; min_base_ce: 0.78500015; 
    FastEstimator-Train: step: 7500; base_ce: 0.7418024; steps/sec: 26.62; 
    FastEstimator-Train: step: 8000; base_ce: 0.60131586; steps/sec: 28.69; 
    FastEstimator-Train: step: 8000; epoch: 8; epoch_time: 36.2 sec; 
    FastEstimator-Eval: step: 8000; epoch: 8; base_ce: 0.8056283; adv_ce: 3.3994384; base accuracy: 0.7374; adversarial accuracy: 0.2362; since_best_base_ce: 1; min_base_ce: 0.78500015; 
    FastEstimator-Train: step: 8500; base_ce: 0.48421046; steps/sec: 26.83; 
    FastEstimator-Train: step: 9000; base_ce: 0.4724892; steps/sec: 28.65; 
    FastEstimator-Train: step: 9000; epoch: 9; epoch_time: 36.1 sec; 
    FastEstimator-Eval: step: 9000; epoch: 9; base_ce: 0.8141311; adv_ce: 3.7335455; base accuracy: 0.7344; adversarial accuracy: 0.2088; since_best_base_ce: 2; min_base_ce: 0.78500015; 
    FastEstimator-Train: step: 9500; base_ce: 0.3733459; steps/sec: 29.25; 
    FastEstimator-Train: step: 10000; base_ce: 0.41766632; steps/sec: 31.09; 
    FastEstimator-Train: step: 10000; epoch: 10; epoch_time: 33.19 sec; 
    FastEstimator-Eval: step: 10000; epoch: 10; base_ce: 0.85542786; adv_ce: 4.0170755; base accuracy: 0.7308; adversarial accuracy: 0.2066; since_best_base_ce: 3; min_base_ce: 0.78500015; 
    FastEstimator-Train: step: 10500; base_ce: 0.41894433; steps/sec: 29.18; 
    FastEstimator-Train: step: 11000; base_ce: 0.65334654; steps/sec: 30.64; 
    FastEstimator-Train: step: 11000; epoch: 11; epoch_time: 33.44 sec; 
    FastEstimator-Eval: step: 11000; epoch: 11; base_ce: 0.9005296; adv_ce: 4.5361137; base accuracy: 0.7234; adversarial accuracy: 0.1788; since_best_base_ce: 4; min_base_ce: 0.78500015; 
    FastEstimator-Train: step: 11500; base_ce: 0.63553435; steps/sec: 29.32; 
    FastEstimator-Train: step: 12000; base_ce: 0.56910044; steps/sec: 31.16; 
    FastEstimator-Train: step: 12000; epoch: 12; epoch_time: 33.12 sec; 
    FastEstimator-Eval: step: 12000; epoch: 12; base_ce: 0.93609977; adv_ce: 4.8335547; base accuracy: 0.7278; adversarial accuracy: 0.1944; since_best_base_ce: 5; min_base_ce: 0.78500015; 
    FastEstimator-Train: step: 12500; base_ce: 0.38925758; steps/sec: 26.87; 
    FastEstimator-Train: step: 13000; base_ce: 0.24855006; steps/sec: 28.67; 
    FastEstimator-Train: step: 13000; epoch: 13; epoch_time: 36.04 sec; 
    FastEstimator-Eval: step: 13000; epoch: 13; base_ce: 1.0094614; adv_ce: 5.402545; base accuracy: 0.7202; adversarial accuracy: 0.182; since_best_base_ce: 6; min_base_ce: 0.78500015; 
    FastEstimator-Train: step: 13500; base_ce: 0.5594125; steps/sec: 26.92; 
    FastEstimator-Train: step: 14000; base_ce: 0.21032642; steps/sec: 28.8; 
    FastEstimator-Train: step: 14000; epoch: 14; epoch_time: 35.93 sec; 
    FastEstimator-Eval: step: 14000; epoch: 14; base_ce: 1.0515163; adv_ce: 5.6984034; base accuracy: 0.7124; adversarial accuracy: 0.1804; since_best_base_ce: 7; min_base_ce: 0.78500015; 
    FastEstimator-Train: step: 14500; base_ce: 0.37550446; steps/sec: 28.22; 
    FastEstimator-Train: step: 15000; base_ce: 0.32693386; steps/sec: 29.72; 
    FastEstimator-Train: step: 15000; epoch: 15; epoch_time: 34.56 sec; 
    FastEstimator-Eval: step: 15000; epoch: 15; base_ce: 1.0898054; adv_ce: 6.124301; base accuracy: 0.7168; adversarial accuracy: 0.1726; since_best_base_ce: 8; min_base_ce: 0.78500015; 
    FastEstimator-Train: step: 15500; base_ce: 0.20683852; steps/sec: 28.59; 
    FastEstimator-Train: step: 16000; base_ce: 0.34564048; steps/sec: 29.86; 
    FastEstimator-Train: step: 16000; epoch: 16; epoch_time: 34.22 sec; 
    FastEstimator-Eval: step: 16000; epoch: 16; base_ce: 1.1524608; adv_ce: 6.8542333; base accuracy: 0.7144; adversarial accuracy: 0.1746; since_best_base_ce: 9; min_base_ce: 0.78500015; 
    FastEstimator-Train: step: 16500; base_ce: 0.2094497; steps/sec: 31.04; 
    FastEstimator-Train: step: 17000; base_ce: 0.15683812; steps/sec: 33.25; 
    FastEstimator-Train: step: 17000; epoch: 17; epoch_time: 31.14 sec; 
    FastEstimator-EarlyStopping: 'base_ce' triggered an early stop. Its best value was 0.7850001454353333 at epoch 7
    FastEstimator-Eval: step: 17000; epoch: 17; base_ce: 1.1774385; adv_ce: 7.219962; base accuracy: 0.7166; adversarial accuracy: 0.1618; since_best_base_ce: 10; min_base_ce: 0.78500015; 
    FastEstimator-BestModelSaver: Restoring model from /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/softmax_best_base_ce.h5
    FastEstimator-Finish: step: 17000; total_time: 742.28 sec; softmax_lr: 0.001; 
    FastEstimator-Test: step: 17000; epoch: 17; base accuracy: 0.7282; adversarial accuracy: 0.234; 



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
    FastEstimator-Train: step: 1; base_ce: 2.2948906; 
    FastEstimator-Train: step: 500; base_ce: 1.8532416; steps/sec: 30.21; 
    FastEstimator-Train: step: 1000; base_ce: 1.7478514; steps/sec: 31.69; 
    FastEstimator-Train: step: 1000; epoch: 1; epoch_time: 33.4 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/ecc_best_base_ce.h5
    FastEstimator-Eval: step: 1000; epoch: 1; base_ce: 1.7138911; adv_ce: 2.1511297; base accuracy: 0.496; adversarial accuracy: 0.3198; since_best_base_ce: 0; min_base_ce: 1.7138911; 
    FastEstimator-Train: step: 1500; base_ce: 1.4022256; steps/sec: 31.97; 
    FastEstimator-Train: step: 2000; base_ce: 1.4052359; steps/sec: 34.93; 
    FastEstimator-Train: step: 2000; epoch: 2; epoch_time: 29.96 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/ecc_best_base_ce.h5
    FastEstimator-Eval: step: 2000; epoch: 2; base_ce: 1.5651137; adv_ce: 2.1517153; base accuracy: 0.5458; adversarial accuracy: 0.3336; since_best_base_ce: 0; min_base_ce: 1.5651137; 
    FastEstimator-Train: step: 2500; base_ce: 1.3534999; steps/sec: 32.84; 
    FastEstimator-Train: step: 3000; base_ce: 1.5177455; steps/sec: 35.16; 
    FastEstimator-Train: step: 3000; epoch: 3; epoch_time: 29.45 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/ecc_best_base_ce.h5
    FastEstimator-Eval: step: 3000; epoch: 3; base_ce: 1.3509419; adv_ce: 2.1408393; base accuracy: 0.6252; adversarial accuracy: 0.3738; since_best_base_ce: 0; min_base_ce: 1.3509419; 
    FastEstimator-Train: step: 3500; base_ce: 1.4140029; steps/sec: 36.55; 
    FastEstimator-Train: step: 4000; base_ce: 1.259077; steps/sec: 38.45; 
    FastEstimator-Train: step: 4000; epoch: 4; epoch_time: 26.68 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/ecc_best_base_ce.h5
    FastEstimator-Eval: step: 4000; epoch: 4; base_ce: 1.2993072; adv_ce: 2.1872435; base accuracy: 0.649; adversarial accuracy: 0.3834; since_best_base_ce: 0; min_base_ce: 1.2993072; 
    FastEstimator-Train: step: 4500; base_ce: 1.146649; steps/sec: 36.42; 
    FastEstimator-Train: step: 5000; base_ce: 0.93379724; steps/sec: 38.36; 
    FastEstimator-Train: step: 5000; epoch: 5; epoch_time: 26.77 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/ecc_best_base_ce.h5
    FastEstimator-Eval: step: 5000; epoch: 5; base_ce: 1.2297044; adv_ce: 2.198256; base accuracy: 0.6682; adversarial accuracy: 0.3808; since_best_base_ce: 0; min_base_ce: 1.2297044; 
    FastEstimator-Train: step: 5500; base_ce: 0.9904622; steps/sec: 36.08; 
    FastEstimator-Train: step: 6000; base_ce: 1.0625097; steps/sec: 36.52; 
    FastEstimator-Train: step: 6000; epoch: 6; epoch_time: 27.55 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/ecc_best_base_ce.h5
    FastEstimator-Eval: step: 6000; epoch: 6; base_ce: 1.2074428; adv_ce: 2.3546593; base accuracy: 0.6862; adversarial accuracy: 0.395; since_best_base_ce: 0; min_base_ce: 1.2074428; 
    FastEstimator-Train: step: 6500; base_ce: 0.7872818; steps/sec: 33.25; 
    FastEstimator-Train: step: 7000; base_ce: 0.88184655; steps/sec: 34.98; 
    FastEstimator-Train: step: 7000; epoch: 7; epoch_time: 29.33 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/ecc_best_base_ce.h5
    FastEstimator-Eval: step: 7000; epoch: 7; base_ce: 1.1496447; adv_ce: 2.326219; base accuracy: 0.699; adversarial accuracy: 0.3872; since_best_base_ce: 0; min_base_ce: 1.1496447; 
    FastEstimator-Train: step: 7500; base_ce: 0.95065016; steps/sec: 33.08; 
    FastEstimator-Train: step: 8000; base_ce: 1.1240141; steps/sec: 37.13; 
    FastEstimator-Train: step: 8000; epoch: 8; epoch_time: 28.59 sec; 
    FastEstimator-Eval: step: 8000; epoch: 8; base_ce: 1.1630411; adv_ce: 2.3640723; base accuracy: 0.6922; adversarial accuracy: 0.3986; since_best_base_ce: 1; min_base_ce: 1.1496447; 
    FastEstimator-Train: step: 8500; base_ce: 1.2048315; steps/sec: 36.49; 
    FastEstimator-Train: step: 9000; base_ce: 1.0226548; steps/sec: 39.1; 
    FastEstimator-Train: step: 9000; epoch: 9; epoch_time: 26.49 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/ecc_best_base_ce.h5
    FastEstimator-Eval: step: 9000; epoch: 9; base_ce: 1.1178105; adv_ce: 2.3132443; base accuracy: 0.706; adversarial accuracy: 0.4152; since_best_base_ce: 0; min_base_ce: 1.1178105; 
    FastEstimator-Train: step: 9500; base_ce: 1.1182468; steps/sec: 36.0; 
    FastEstimator-Train: step: 10000; base_ce: 0.6692859; steps/sec: 37.89; 
    FastEstimator-Train: step: 10000; epoch: 10; epoch_time: 27.08 sec; 
    FastEstimator-Eval: step: 10000; epoch: 10; base_ce: 1.1335105; adv_ce: 2.4508705; base accuracy: 0.7068; adversarial accuracy: 0.4008; since_best_base_ce: 1; min_base_ce: 1.1178105; 
    FastEstimator-Train: step: 10500; base_ce: 0.700335; steps/sec: 35.91; 
    FastEstimator-Train: step: 11000; base_ce: 0.74321246; steps/sec: 38.14; 
    FastEstimator-Train: step: 11000; epoch: 11; epoch_time: 27.03 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/ecc_best_base_ce.h5
    FastEstimator-Eval: step: 11000; epoch: 11; base_ce: 1.077382; adv_ce: 2.298885; base accuracy: 0.7224; adversarial accuracy: 0.4282; since_best_base_ce: 0; min_base_ce: 1.077382; 
    FastEstimator-Train: step: 11500; base_ce: 0.7760241; steps/sec: 36.38; 
    FastEstimator-Train: step: 12000; base_ce: 0.53444064; steps/sec: 38.03; 
    FastEstimator-Train: step: 12000; epoch: 12; epoch_time: 26.89 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/ecc_best_base_ce.h5
    FastEstimator-Eval: step: 12000; epoch: 12; base_ce: 1.0708356; adv_ce: 2.4382222; base accuracy: 0.7252; adversarial accuracy: 0.4212; since_best_base_ce: 0; min_base_ce: 1.0708356; 
    FastEstimator-Train: step: 12500; base_ce: 0.6889807; steps/sec: 36.63; 
    FastEstimator-Train: step: 13000; base_ce: 0.9038168; steps/sec: 38.36; 
    FastEstimator-Train: step: 13000; epoch: 13; epoch_time: 26.69 sec; 
    FastEstimator-Eval: step: 13000; epoch: 13; base_ce: 1.109232; adv_ce: 2.4642434; base accuracy: 0.7188; adversarial accuracy: 0.4212; since_best_base_ce: 1; min_base_ce: 1.0708356; 
    FastEstimator-Train: step: 13500; base_ce: 1.019563; steps/sec: 39.41; 
    FastEstimator-Train: step: 14000; base_ce: 0.62402534; steps/sec: 42.24; 
    FastEstimator-Train: step: 14000; epoch: 14; epoch_time: 24.52 sec; 
    FastEstimator-Eval: step: 14000; epoch: 14; base_ce: 1.0935918; adv_ce: 2.4698822; base accuracy: 0.7236; adversarial accuracy: 0.4312; since_best_base_ce: 2; min_base_ce: 1.0708356; 
    FastEstimator-Train: step: 14500; base_ce: 0.8987514; steps/sec: 39.77; 
    FastEstimator-Train: step: 15000; base_ce: 0.7350322; steps/sec: 39.5; 
    FastEstimator-Train: step: 15000; epoch: 15; epoch_time: 25.24 sec; 
    FastEstimator-Eval: step: 15000; epoch: 15; base_ce: 1.1243794; adv_ce: 2.4714823; base accuracy: 0.7138; adversarial accuracy: 0.4264; since_best_base_ce: 3; min_base_ce: 1.0708356; 
    FastEstimator-Train: step: 15500; base_ce: 0.6691685; steps/sec: 35.86; 
    FastEstimator-Train: step: 16000; base_ce: 0.7693179; steps/sec: 38.54; 
    FastEstimator-Train: step: 16000; epoch: 16; epoch_time: 26.92 sec; 
    FastEstimator-Eval: step: 16000; epoch: 16; base_ce: 1.1038796; adv_ce: 2.5672605; base accuracy: 0.7206; adversarial accuracy: 0.414; since_best_base_ce: 4; min_base_ce: 1.0708356; 
    FastEstimator-Train: step: 16500; base_ce: 0.55752677; steps/sec: 36.4; 
    FastEstimator-Train: step: 17000; base_ce: 0.95395017; steps/sec: 38.6; 
    FastEstimator-Train: step: 17000; epoch: 17; epoch_time: 26.68 sec; 
    FastEstimator-Eval: step: 17000; epoch: 17; base_ce: 1.0992229; adv_ce: 2.4245234; base accuracy: 0.726; adversarial accuracy: 0.4484; since_best_base_ce: 5; min_base_ce: 1.0708356; 
    FastEstimator-Train: step: 17500; base_ce: 0.97094285; steps/sec: 36.61; 
    FastEstimator-Train: step: 18000; base_ce: 0.6378223; steps/sec: 39.16; 
    FastEstimator-Train: step: 18000; epoch: 18; epoch_time: 26.43 sec; 
    FastEstimator-Eval: step: 18000; epoch: 18; base_ce: 1.1157478; adv_ce: 2.496202; base accuracy: 0.719; adversarial accuracy: 0.4378; since_best_base_ce: 6; min_base_ce: 1.0708356; 
    FastEstimator-Train: step: 18500; base_ce: 0.6535682; steps/sec: 36.22; 
    FastEstimator-Train: step: 19000; base_ce: 0.7969613; steps/sec: 38.31; 
    FastEstimator-Train: step: 19000; epoch: 19; epoch_time: 26.86 sec; 
    FastEstimator-Eval: step: 19000; epoch: 19; base_ce: 1.0798271; adv_ce: 2.382594; base accuracy: 0.7284; adversarial accuracy: 0.4556; since_best_base_ce: 7; min_base_ce: 1.0708356; 
    FastEstimator-Train: step: 19500; base_ce: 0.3818073; steps/sec: 36.99; 
    FastEstimator-Train: step: 20000; base_ce: 0.6504534; steps/sec: 38.94; 
    FastEstimator-Train: step: 20000; epoch: 20; epoch_time: 26.36 sec; 
    FastEstimator-Eval: step: 20000; epoch: 20; base_ce: 1.0884305; adv_ce: 2.4666903; base accuracy: 0.7278; adversarial accuracy: 0.4482; since_best_base_ce: 8; min_base_ce: 1.0708356; 
    FastEstimator-Train: step: 20500; base_ce: 0.644705; steps/sec: 37.49; 
    FastEstimator-Train: step: 21000; base_ce: 0.6375694; steps/sec: 41.72; 
    FastEstimator-Train: step: 21000; epoch: 21; epoch_time: 25.32 sec; 
    FastEstimator-Eval: step: 21000; epoch: 21; base_ce: 1.1050826; adv_ce: 2.4939957; base accuracy: 0.7278; adversarial accuracy: 0.4492; since_best_base_ce: 9; min_base_ce: 1.0708356; 
    FastEstimator-Train: step: 21500; base_ce: 0.7648297; steps/sec: 39.6; 
    FastEstimator-Train: step: 22000; base_ce: 0.64078635; steps/sec: 42.45; 
    FastEstimator-Train: step: 22000; epoch: 22; epoch_time: 24.41 sec; 
    FastEstimator-EarlyStopping: 'base_ce' triggered an early stop. Its best value was 1.0708355903625488 at epoch 12
    FastEstimator-Eval: step: 22000; epoch: 22; base_ce: 1.1157986; adv_ce: 2.3670042; base accuracy: 0.7294; adversarial accuracy: 0.4818; since_best_base_ce: 10; min_base_ce: 1.0708356; 
    FastEstimator-BestModelSaver: Restoring model from /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/ecc_best_base_ce.h5
    FastEstimator-Finish: step: 22000; total_time: 694.57 sec; ecc_lr: 0.001; 
    FastEstimator-Test: step: 22000; epoch: 22; base accuracy: 0.7298; adversarial accuracy: 0.4174; 



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
    FastEstimator-Train: step: 1; base_ce: 2.3076158; 
    FastEstimator-Train: step: 500; base_ce: 1.7894692; steps/sec: 39.9; 
    FastEstimator-Train: step: 1000; base_ce: 1.6115338; steps/sec: 41.67; 
    FastEstimator-Train: step: 1000; epoch: 1; epoch_time: 25.65 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 1000; epoch: 1; base_ce: 1.772432; adv_ce: 2.2166047; base accuracy: 0.4562; adversarial accuracy: 0.3138; since_best_base_ce: 0; min_base_ce: 1.772432; 
    FastEstimator-Train: step: 1500; base_ce: 1.6917462; steps/sec: 39.34; 
    FastEstimator-Train: step: 2000; base_ce: 1.5339103; steps/sec: 41.41; 
    FastEstimator-Train: step: 2000; epoch: 2; epoch_time: 24.79 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 2000; epoch: 2; base_ce: 1.53139; adv_ce: 2.1004038; base accuracy: 0.5708; adversarial accuracy: 0.3744; since_best_base_ce: 0; min_base_ce: 1.53139; 
    FastEstimator-Train: step: 2500; base_ce: 1.65984; steps/sec: 39.34; 
    FastEstimator-Train: step: 3000; base_ce: 1.5187409; steps/sec: 41.11; 
    FastEstimator-Train: step: 3000; epoch: 3; epoch_time: 24.87 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 3000; epoch: 3; base_ce: 1.407918; adv_ce: 2.1317058; base accuracy: 0.6104; adversarial accuracy: 0.3704; since_best_base_ce: 0; min_base_ce: 1.407918; 
    FastEstimator-Train: step: 3500; base_ce: 1.4999541; steps/sec: 36.05; 
    FastEstimator-Train: step: 4000; base_ce: 1.3882403; steps/sec: 38.61; 
    FastEstimator-Train: step: 4000; epoch: 4; epoch_time: 26.82 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 4000; epoch: 4; base_ce: 1.3866385; adv_ce: 2.1057322; base accuracy: 0.6182; adversarial accuracy: 0.3706; since_best_base_ce: 0; min_base_ce: 1.3866385; 
    FastEstimator-Train: step: 4500; base_ce: 1.4213487; steps/sec: 35.65; 
    FastEstimator-Train: step: 5000; base_ce: 1.1099585; steps/sec: 38.0; 
    FastEstimator-Train: step: 5000; epoch: 5; epoch_time: 27.18 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 5000; epoch: 5; base_ce: 1.285622; adv_ce: 2.1949887; base accuracy: 0.6532; adversarial accuracy: 0.3748; since_best_base_ce: 0; min_base_ce: 1.285622; 
    FastEstimator-Train: step: 5500; base_ce: 1.0962647; steps/sec: 37.13; 
    FastEstimator-Train: step: 6000; base_ce: 1.19081; steps/sec: 41.86; 
    FastEstimator-Train: step: 6000; epoch: 6; epoch_time: 25.41 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 6000; epoch: 6; base_ce: 1.2323712; adv_ce: 2.191548; base accuracy: 0.6758; adversarial accuracy: 0.3874; since_best_base_ce: 0; min_base_ce: 1.2323712; 
    FastEstimator-Train: step: 6500; base_ce: 1.0899575; steps/sec: 39.26; 
    FastEstimator-Train: step: 7000; base_ce: 1.0509219; steps/sec: 40.43; 
    FastEstimator-Train: step: 7000; epoch: 7; epoch_time: 25.11 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 7000; epoch: 7; base_ce: 1.1828624; adv_ce: 2.2274768; base accuracy: 0.691; adversarial accuracy: 0.3988; since_best_base_ce: 0; min_base_ce: 1.1828624; 
    FastEstimator-Train: step: 7500; base_ce: 1.1362395; steps/sec: 38.76; 
    FastEstimator-Train: step: 8000; base_ce: 1.1766993; steps/sec: 40.36; 
    FastEstimator-Train: step: 8000; epoch: 8; epoch_time: 25.28 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 8000; epoch: 8; base_ce: 1.179946; adv_ce: 2.2246737; base accuracy: 0.6894; adversarial accuracy: 0.4022; since_best_base_ce: 0; min_base_ce: 1.179946; 
    FastEstimator-Train: step: 8500; base_ce: 1.06356; steps/sec: 39.28; 
    FastEstimator-Train: step: 9000; base_ce: 1.0581198; steps/sec: 41.16; 
    FastEstimator-Train: step: 9000; epoch: 9; epoch_time: 24.9 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 9000; epoch: 9; base_ce: 1.1292516; adv_ce: 2.193486; base accuracy: 0.7054; adversarial accuracy: 0.4132; since_best_base_ce: 0; min_base_ce: 1.1292516; 
    FastEstimator-Train: step: 9500; base_ce: 1.0268096; steps/sec: 38.59; 
    FastEstimator-Train: step: 10000; base_ce: 1.2578716; steps/sec: 40.34; 
    FastEstimator-Train: step: 10000; epoch: 10; epoch_time: 25.34 sec; 
    FastEstimator-Eval: step: 10000; epoch: 10; base_ce: 1.1553321; adv_ce: 2.3133547; base accuracy: 0.6994; adversarial accuracy: 0.4076; since_best_base_ce: 1; min_base_ce: 1.1292516; 
    FastEstimator-Train: step: 10500; base_ce: 0.9665863; steps/sec: 35.17; 
    FastEstimator-Train: step: 11000; base_ce: 0.9702922; steps/sec: 37.54; 
    FastEstimator-Train: step: 11000; epoch: 11; epoch_time: 27.54 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 11000; epoch: 11; base_ce: 1.0691696; adv_ce: 2.2379944; base accuracy: 0.7196; adversarial accuracy: 0.4186; since_best_base_ce: 0; min_base_ce: 1.0691696; 
    FastEstimator-Train: step: 11500; base_ce: 1.0323644; steps/sec: 35.62; 
    FastEstimator-Train: step: 12000; base_ce: 0.7866042; steps/sec: 40.58; 
    FastEstimator-Train: step: 12000; epoch: 12; epoch_time: 26.35 sec; 
    FastEstimator-Eval: step: 12000; epoch: 12; base_ce: 1.0878896; adv_ce: 2.3281896; base accuracy: 0.7218; adversarial accuracy: 0.4046; since_best_base_ce: 1; min_base_ce: 1.0691696; 
    FastEstimator-Train: step: 12500; base_ce: 0.80621284; steps/sec: 37.97; 
    FastEstimator-Train: step: 13000; base_ce: 0.8485186; steps/sec: 38.49; 
    FastEstimator-Train: step: 13000; epoch: 13; epoch_time: 26.18 sec; 
    FastEstimator-Eval: step: 13000; epoch: 13; base_ce: 1.1110873; adv_ce: 2.3123732; base accuracy: 0.7134; adversarial accuracy: 0.428; since_best_base_ce: 2; min_base_ce: 1.0691696; 
    FastEstimator-Train: step: 13500; base_ce: 0.6478882; steps/sec: 39.09; 
    FastEstimator-Train: step: 14000; base_ce: 0.9344132; steps/sec: 41.71; 
    FastEstimator-Train: step: 14000; epoch: 14; epoch_time: 24.77 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 14000; epoch: 14; base_ce: 1.0669813; adv_ce: 2.3520224; base accuracy: 0.7282; adversarial accuracy: 0.4164; since_best_base_ce: 0; min_base_ce: 1.0669813; 
    FastEstimator-Train: step: 14500; base_ce: 0.7195329; steps/sec: 39.72; 
    FastEstimator-Train: step: 15000; base_ce: 1.0449741; steps/sec: 42.56; 
    FastEstimator-Train: step: 15000; epoch: 15; epoch_time: 24.33 sec; 
    FastEstimator-Eval: step: 15000; epoch: 15; base_ce: 1.0872117; adv_ce: 2.3199148; base accuracy: 0.7226; adversarial accuracy: 0.4142; since_best_base_ce: 1; min_base_ce: 1.0669813; 
    FastEstimator-Train: step: 15500; base_ce: 0.8833719; steps/sec: 40.01; 
    FastEstimator-Train: step: 16000; base_ce: 1.0172313; steps/sec: 41.95; 
    FastEstimator-Train: step: 16000; epoch: 16; epoch_time: 24.42 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 16000; epoch: 16; base_ce: 1.0428473; adv_ce: 2.3096952; base accuracy: 0.732; adversarial accuracy: 0.434; since_best_base_ce: 0; min_base_ce: 1.0428473; 
    FastEstimator-Train: step: 16500; base_ce: 0.9528541; steps/sec: 38.97; 
    FastEstimator-Train: step: 17000; base_ce: 0.6921418; steps/sec: 41.57; 
    FastEstimator-Train: step: 17000; epoch: 17; epoch_time: 24.87 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 17000; epoch: 17; base_ce: 1.0407811; adv_ce: 2.3547754; base accuracy: 0.7322; adversarial accuracy: 0.4232; since_best_base_ce: 0; min_base_ce: 1.0407811; 
    FastEstimator-Train: step: 17500; base_ce: 0.85563445; steps/sec: 39.17; 
    FastEstimator-Train: step: 18000; base_ce: 0.84442306; steps/sec: 41.73; 
    FastEstimator-Train: step: 18000; epoch: 18; epoch_time: 24.74 sec; 
    FastEstimator-Eval: step: 18000; epoch: 18; base_ce: 1.067384; adv_ce: 2.3860226; base accuracy: 0.7322; adversarial accuracy: 0.4146; since_best_base_ce: 1; min_base_ce: 1.0407811; 
    FastEstimator-Train: step: 18500; base_ce: 0.6891441; steps/sec: 39.71; 
    FastEstimator-Train: step: 19000; base_ce: 0.4970177; steps/sec: 41.33; 
    FastEstimator-Train: step: 19000; epoch: 19; epoch_time: 24.69 sec; 
    FastEstimator-Eval: step: 19000; epoch: 19; base_ce: 1.0554608; adv_ce: 2.3364587; base accuracy: 0.7318; adversarial accuracy: 0.4246; since_best_base_ce: 2; min_base_ce: 1.0407811; 
    FastEstimator-Train: step: 19500; base_ce: 0.845229; steps/sec: 36.31; 
    FastEstimator-Train: step: 20000; base_ce: 0.7282557; steps/sec: 37.79; 
    FastEstimator-Train: step: 20000; epoch: 20; epoch_time: 27.0 sec; 
    FastEstimator-Eval: step: 20000; epoch: 20; base_ce: 1.0579226; adv_ce: 2.3450043; base accuracy: 0.73; adversarial accuracy: 0.4272; since_best_base_ce: 3; min_base_ce: 1.0407811; 
    FastEstimator-Train: step: 20500; base_ce: 0.6578657; steps/sec: 35.22; 
    FastEstimator-Train: step: 21000; base_ce: 1.133787; steps/sec: 40.44; 
    FastEstimator-Train: step: 21000; epoch: 21; epoch_time: 26.58 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Eval: step: 21000; epoch: 21; base_ce: 1.0174779; adv_ce: 2.310723; base accuracy: 0.7404; adversarial accuracy: 0.4388; since_best_base_ce: 0; min_base_ce: 1.0174779; 
    FastEstimator-Train: step: 21500; base_ce: 0.8653169; steps/sec: 39.44; 
    FastEstimator-Train: step: 22000; base_ce: 0.661098; steps/sec: 42.19; 
    FastEstimator-Train: step: 22000; epoch: 22; epoch_time: 24.51 sec; 
    FastEstimator-Eval: step: 22000; epoch: 22; base_ce: 1.0703703; adv_ce: 2.414565; base accuracy: 0.7312; adversarial accuracy: 0.4176; since_best_base_ce: 1; min_base_ce: 1.0174779; 
    FastEstimator-Train: step: 22500; base_ce: 0.89490587; steps/sec: 38.64; 
    FastEstimator-Train: step: 23000; base_ce: 0.9243805; steps/sec: 40.36; 
    FastEstimator-Train: step: 23000; epoch: 23; epoch_time: 25.34 sec; 
    FastEstimator-Eval: step: 23000; epoch: 23; base_ce: 1.0419381; adv_ce: 2.3707998; base accuracy: 0.7382; adversarial accuracy: 0.4332; since_best_base_ce: 2; min_base_ce: 1.0174779; 
    FastEstimator-Train: step: 23500; base_ce: 0.37459114; steps/sec: 39.22; 
    FastEstimator-Train: step: 24000; base_ce: 0.77817583; steps/sec: 39.84; 
    FastEstimator-Train: step: 24000; epoch: 24; epoch_time: 25.29 sec; 
    FastEstimator-Eval: step: 24000; epoch: 24; base_ce: 1.0574633; adv_ce: 2.3102021; base accuracy: 0.7332; adversarial accuracy: 0.4494; since_best_base_ce: 3; min_base_ce: 1.0174779; 
    FastEstimator-Train: step: 24500; base_ce: 0.7001125; steps/sec: 38.9; 
    FastEstimator-Train: step: 25000; base_ce: 0.6399208; steps/sec: 37.16; 
    FastEstimator-Train: step: 25000; epoch: 25; epoch_time: 26.3 sec; 
    FastEstimator-Eval: step: 25000; epoch: 25; base_ce: 1.0435; adv_ce: 2.3095825; base accuracy: 0.7404; adversarial accuracy: 0.446; since_best_base_ce: 4; min_base_ce: 1.0174779; 
    FastEstimator-Train: step: 25500; base_ce: 0.6391239; steps/sec: 38.9; 
    FastEstimator-Train: step: 26000; base_ce: 0.91452485; steps/sec: 41.62; 
    FastEstimator-Train: step: 26000; epoch: 26; epoch_time: 24.87 sec; 
    FastEstimator-Eval: step: 26000; epoch: 26; base_ce: 1.0903724; adv_ce: 2.3547668; base accuracy: 0.7284; adversarial accuracy: 0.4442; since_best_base_ce: 5; min_base_ce: 1.0174779; 
    FastEstimator-Train: step: 26500; base_ce: 0.97995234; steps/sec: 39.34; 
    FastEstimator-Train: step: 27000; base_ce: 0.7880356; steps/sec: 39.81; 
    FastEstimator-Train: step: 27000; epoch: 27; epoch_time: 25.28 sec; 
    FastEstimator-Eval: step: 27000; epoch: 27; base_ce: 1.04056; adv_ce: 2.3332062; base accuracy: 0.747; adversarial accuracy: 0.448; since_best_base_ce: 6; min_base_ce: 1.0174779; 
    FastEstimator-Train: step: 27500; base_ce: 0.7402453; steps/sec: 38.51; 
    FastEstimator-Train: step: 28000; base_ce: 0.5625182; steps/sec: 39.95; 
    FastEstimator-Train: step: 28000; epoch: 28; epoch_time: 25.49 sec; 
    FastEstimator-Eval: step: 28000; epoch: 28; base_ce: 1.0731995; adv_ce: 2.4006343; base accuracy: 0.7344; adversarial accuracy: 0.4334; since_best_base_ce: 7; min_base_ce: 1.0174779; 
    FastEstimator-Train: step: 28500; base_ce: 0.6383535; steps/sec: 39.32; 
    FastEstimator-Train: step: 29000; base_ce: 0.842826; steps/sec: 40.93; 
    FastEstimator-Train: step: 29000; epoch: 29; epoch_time: 24.94 sec; 
    FastEstimator-Eval: step: 29000; epoch: 29; base_ce: 1.0614614; adv_ce: 2.3708925; base accuracy: 0.7382; adversarial accuracy: 0.4478; since_best_base_ce: 8; min_base_ce: 1.0174779; 
    FastEstimator-Train: step: 29500; base_ce: 0.62300307; steps/sec: 38.82; 
    FastEstimator-Train: step: 30000; base_ce: 0.7383355; steps/sec: 41.19; 
    FastEstimator-Train: step: 30000; epoch: 30; epoch_time: 25.03 sec; 
    FastEstimator-Eval: step: 30000; epoch: 30; base_ce: 1.0614293; adv_ce: 2.2735631; base accuracy: 0.7356; adversarial accuracy: 0.4646; since_best_base_ce: 9; min_base_ce: 1.0174779; 
    FastEstimator-Train: step: 30500; base_ce: 0.61352885; steps/sec: 38.38; 
    FastEstimator-Train: step: 31000; base_ce: 0.41669288; steps/sec: 40.06; 
    FastEstimator-Train: step: 31000; epoch: 31; epoch_time: 25.51 sec; 
    FastEstimator-EarlyStopping: 'base_ce' triggered an early stop. Its best value was 1.0174778699874878 at epoch 21
    FastEstimator-Eval: step: 31000; epoch: 31; base_ce: 1.0607773; adv_ce: 2.2538073; base accuracy: 0.7358; adversarial accuracy: 0.476; since_best_base_ce: 10; min_base_ce: 1.0174779; 
    FastEstimator-BestModelSaver: Restoring model from /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp__tav_q_/hydra_ecc_best_base_ce.h5
    FastEstimator-Finish: step: 31000; total_time: 914.06 sec; hydra_ecc_lr: 0.001; 
    FastEstimator-Test: step: 31000; epoch: 31; base accuracy: 0.7326; adversarial accuracy: 0.4258; 


## Comparing the Results


```python
logs = visualize_logs(experiments=[softmax_results, ecc_results, hydra_results], ignore_metrics={'ecc_lr', 'hydra_ecc_lr', 'softmax_lr', 'logging_interval', 'num_device', 'epoch_time', 'min_base_ce', 'adv_ce', 'total_time'})
```


    
![png](assets/branches/master/example/adversarial_training/ecc_files/ecc_22_0.png)
    


As you can see, the conventional network using softmax to convert logits to class probabilities actually gets more and more vulnerable to adversarial attacks as training progresses. It also quickly overfits to the data, reaching an optimal performance around epoch 7. By simply switching the softmax layer for an error-correcting-code, the network is able to train for around 16 epochs before starting to cap out, and even then continuing to train it results in better and better adversarial performance. Creating a multi-headed ecc output layer allows still more training and higher peak performances. If you were to run the experiment out to 160 epochs you would find that the adversarial accuracy can reach between 60-70% with only a slight accuracy degradation on clean samples (performance still above 70%). This is significantly better performance than networks trained specifically to combat this attack, shown in the [FGSM](../fgsm/fgsm.ipynb) notebook. Note also that their is virtually no additional cost to training using ECC as opposed to softmax in terms of steps/sec. This is a big benefit over FGSM, where the training time for each step is doubled. With these benefits in mind, you may want to consider never using softmax again.
