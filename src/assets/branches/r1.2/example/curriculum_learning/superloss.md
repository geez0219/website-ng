# Curriculum Learning with SuperLoss (Tensorflow Backend)
In this example, we are going to demonstrate how to easily add curriculum learning to any project using SuperLoss (paper available [here](https://papers.nips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf)). When humans learn something in school, we are first taught how to do easy versions of the task before graduating to more difficult problems. Curriculum learning seeks to emulate that process with neural networks. One way to do this would be to try and modify a data pipeline to change the order in which it presents examples, but an easier way is to simply modify your loss term to reduce the contribution of difficult examples until later on during training. Curriculum learning has been shown to be especially useful when you have label noise in your dataset, since noisy samples are essentially 'hard' and you want to put off trying to learn them. 

## Import the required libraries


```python
import math
import tempfile

import numpy as np
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.python.keras.models import Sequential

import fastestimator as fe
from fastestimator.dataset.data import cifair100
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy, SuperLoss
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import MCC
from fastestimator.trace.xai import LabelTracker
```


```python
#training parameters
epochs = 50
batch_size = 128
max_train_steps_per_epoch = None
max_eval_steps_per_epoch = None
save_dir = tempfile.mkdtemp()
```

## Step 1 - Data preparation

In this step, we will load the [ciFAIR100](https://arxiv.org/pdf/1902.00423.pdf) training and validation datasets. We use a FastEstimator API to load the dataset and then get a test set by splitting 50% of the data off of the evaluation set. We are also going to corrupt the training data by adding 40% label noise, to simulate the fact that many real-world datasets may have low quality annotations.


```python
from fastestimator.dataset.data import cifair100

train_data, eval_data = cifair100.load_data()
test_data = eval_data.split(0.5)

def corrupt_dataset(dataset, n_classes=100, corruption_fraction=0.4):
    # Keep track of which samples were corrupted for visualization later
    corrupted = [0 for _ in range(len(dataset))]
    # Perform the actual label corruption
    n_samples_per_class = len(dataset) // n_classes
    n_to_corrupt_per_class = math.floor(corruption_fraction * n_samples_per_class)
    n_corrupted = [0] * n_classes
    i = 0
    while any([elem < n_to_corrupt_per_class for elem in n_corrupted]):
        current_class = dataset[i]['y'].item()
        if n_corrupted[current_class] < n_to_corrupt_per_class:
            dataset[i]['y'] = (dataset[i]['y'] + np.random.randint(1, n_classes)) % n_classes
            n_corrupted[current_class] += 1
            corrupted[i] = 1
        i += 1
    # Put the corruption labels into the dataset for visualization
    dataset['data_labels'] = np.array(corrupted, dtype=np.int).reshape((len(dataset), 1))

corrupt_dataset(train_data)
```

## Step 2 - Build some Estimators

We will define a function that builds relatively simple estimators given only a particular loss function as an input. We can then compare the effects of using a regular loss versus a SuperLoss on our artificially corrupted dataset.


```python
def big_lenet(classes=100, input_shape=(32, 32, 3)):
    # Like a LeNet model, but bigger. 
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='swish', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='swish'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='swish'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='swish'))
    model.add(BatchNormalization())
    model.add(Dense(classes, activation='softmax'))
    return model

def build_estimator(loss_op):
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           test_data=test_data,
                           batch_size=batch_size,
                           ops=[Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
                                PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
                                RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
                                Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
                                CoarseDropout(inputs="x", outputs="x", max_holes=1, mode="train"),
                                ])
    model = fe.build(model_fn=big_lenet, optimizer_fn='adam')
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        loss_op,  # <<<----------------------------- This is where the secret sauce will go
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        MCC(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="mcc", save_best_mode="max", load_best_final=True),
        # We will also visualize the difference between the normal and corrupted image confidence scores. You could follow this with an
        # ImageViewer trace, but we will get the data out of the system summary instead later for viewing.
        LabelTracker(metric="confidence", label="data_labels", label_mapping={"Normal": 0, "Corrupted": 1}, mode="train", outputs="label_confidence"),
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch,
                             log_steps=300)
    return estimator
```

## Step 3 - Train a baseline model with a regular loss

Let's start by training a regular model using standard CrossEntropy and see what we get. We will also define a fake SuperLoss wrapper to get sample confidence estimates in order to visualize the differences between clean and corrupted data performance.


```python
class FakeSuperLoss(SuperLoss):
    def forward(self, data, state):
        superloss, confidence = super().forward(data, state)
        regularloss = fe.backend.reduce_mean(self.loss.forward(data, state))
        return [regularloss, confidence]

loss = FakeSuperLoss(CrossEntropy(inputs=("y_pred", "y"), outputs="ce"), output_confidence="confidence")
estimator_regular = build_estimator(loss)
regular = estimator_regular.fit("RegularLoss")
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; logging_interval: 300; num_device: 0;
    FastEstimator-Train: step: 1; ce: 5.2058554;
    FastEstimator-Train: step: 300; ce: 4.219718; steps/sec: 10.95;
    FastEstimator-Train: step: 391; epoch: 1; epoch_time: 38.74 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 391; epoch: 1; ce: 3.9026656; max_mcc: 0.12934212333991163; mcc: 0.12934212333991163; since_best_mcc: 0;
    FastEstimator-Train: step: 600; ce: 4.3571787; steps/sec: 11.08;
    FastEstimator-Train: step: 782; epoch: 2; epoch_time: 32.28 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 782; epoch: 2; ce: 3.5337455; max_mcc: 0.18944058177492504; mcc: 0.18944058177492504; since_best_mcc: 0;
    FastEstimator-Train: step: 900; ce: 4.117856; steps/sec: 12.37;
    FastEstimator-Train: step: 1173; epoch: 3; epoch_time: 30.05 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 1173; epoch: 3; ce: 3.2818522; max_mcc: 0.2296687999203852; mcc: 0.2296687999203852; since_best_mcc: 0;
    FastEstimator-Train: step: 1200; ce: 3.9846687; steps/sec: 12.98;
    FastEstimator-Train: step: 1500; ce: 4.0104823; steps/sec: 13.25;
    FastEstimator-Train: step: 1564; epoch: 4; epoch_time: 29.78 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 1564; epoch: 4; ce: 3.1701896; max_mcc: 0.25955912897520683; mcc: 0.25955912897520683; since_best_mcc: 0;
    FastEstimator-Train: step: 1800; ce: 3.9537716; steps/sec: 13.6;
    FastEstimator-Train: step: 1955; epoch: 5; epoch_time: 28.67 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 1955; epoch: 5; ce: 3.0616918; max_mcc: 0.2738517571518162; mcc: 0.2738517571518162; since_best_mcc: 0;
    FastEstimator-Train: step: 2100; ce: 3.7930996; steps/sec: 12.33;
    FastEstimator-Train: step: 2346; epoch: 6; epoch_time: 35.36 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 2346; epoch: 6; ce: 3.0023494; max_mcc: 0.29504850594691345; mcc: 0.29504850594691345; since_best_mcc: 0;
    FastEstimator-Train: step: 2400; ce: 3.8626804; steps/sec: 11.01;
    FastEstimator-Train: step: 2700; ce: 3.7470045; steps/sec: 11.03;
    FastEstimator-Train: step: 2737; epoch: 7; epoch_time: 35.26 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 2737; epoch: 7; ce: 2.9440446; max_mcc: 0.31184990626643844; mcc: 0.31184990626643844; since_best_mcc: 0;
    FastEstimator-Train: step: 3000; ce: 3.8534527; steps/sec: 12.8;
    FastEstimator-Train: step: 3128; epoch: 8; epoch_time: 31.79 sec;
    FastEstimator-Eval: step: 3128; epoch: 8; ce: 2.9298024; max_mcc: 0.31184990626643844; mcc: 0.30966230920446447; since_best_mcc: 1;
    FastEstimator-Train: step: 3300; ce: 3.826063; steps/sec: 11.21;
    FastEstimator-Train: step: 3519; epoch: 9; epoch_time: 33.53 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 3519; epoch: 9; ce: 2.8336983; max_mcc: 0.3276114092420876; mcc: 0.3276114092420876; since_best_mcc: 0;
    FastEstimator-Train: step: 3600; ce: 3.6307273; steps/sec: 11.71;
    FastEstimator-Train: step: 3900; ce: 4.151486; steps/sec: 12.79;
    FastEstimator-Train: step: 3910; epoch: 10; epoch_time: 31.58 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 3910; epoch: 10; ce: 2.8240047; max_mcc: 0.33407211926037067; mcc: 0.33407211926037067; since_best_mcc: 0;
    FastEstimator-Train: step: 4200; ce: 3.6304643; steps/sec: 13.55;
    FastEstimator-Train: step: 4301; epoch: 11; epoch_time: 28.67 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 4301; epoch: 11; ce: 2.8075173; max_mcc: 0.33959039323223245; mcc: 0.33959039323223245; since_best_mcc: 0;
    FastEstimator-Train: step: 4500; ce: 3.5259557; steps/sec: 12.89;
    FastEstimator-Train: step: 4692; epoch: 12; epoch_time: 31.05 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 4692; epoch: 12; ce: 2.7678986; max_mcc: 0.3451374939289194; mcc: 0.3451374939289194; since_best_mcc: 0;
    FastEstimator-Train: step: 4800; ce: 3.729338; steps/sec: 12.65;
    FastEstimator-Train: step: 5083; epoch: 13; epoch_time: 29.38 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 5083; epoch: 13; ce: 2.746238; max_mcc: 0.35493389750810334; mcc: 0.35493389750810334; since_best_mcc: 0;
    FastEstimator-Train: step: 5100; ce: 3.599938; steps/sec: 13.41;
    FastEstimator-Train: step: 5400; ce: 3.8274336; steps/sec: 12.81;
    FastEstimator-Train: step: 5474; epoch: 14; epoch_time: 30.88 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 5474; epoch: 14; ce: 2.6945357; max_mcc: 0.36127521296768766; mcc: 0.36127521296768766; since_best_mcc: 0;
    FastEstimator-Train: step: 5700; ce: 3.7140992; steps/sec: 12.63;
    FastEstimator-Train: step: 5865; epoch: 15; epoch_time: 30.1 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 5865; epoch: 15; ce: 2.6803436; max_mcc: 0.366931886211223; mcc: 0.366931886211223; since_best_mcc: 0;
    FastEstimator-Train: step: 6000; ce: 3.7583175; steps/sec: 12.82;
    FastEstimator-Train: step: 6256; epoch: 16; epoch_time: 32.22 sec;
    FastEstimator-Eval: step: 6256; epoch: 16; ce: 2.7126276; max_mcc: 0.366931886211223; mcc: 0.3537003555244694; since_best_mcc: 1;
    FastEstimator-Train: step: 6300; ce: 3.4689145; steps/sec: 12.0;
    FastEstimator-Train: step: 6600; ce: 3.5330806; steps/sec: 12.82;
    FastEstimator-Train: step: 6647; epoch: 17; epoch_time: 30.59 sec;
    FastEstimator-Eval: step: 6647; epoch: 17; ce: 2.6502252; max_mcc: 0.366931886211223; mcc: 0.3651458442654979; since_best_mcc: 2;
    FastEstimator-Train: step: 6900; ce: 3.4862967; steps/sec: 13.3;
    FastEstimator-Train: step: 7038; epoch: 18; epoch_time: 29.59 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 7038; epoch: 18; ce: 2.6689048; max_mcc: 0.36889325117891375; mcc: 0.36889325117891375; since_best_mcc: 0;
    FastEstimator-Train: step: 7200; ce: 3.697967; steps/sec: 13.52;
    FastEstimator-Train: step: 7429; epoch: 19; epoch_time: 28.55 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 7429; epoch: 19; ce: 2.6127114; max_mcc: 0.37981499872517466; mcc: 0.37981499872517466; since_best_mcc: 0;
    FastEstimator-Train: step: 7500; ce: 3.566219; steps/sec: 13.31;
    FastEstimator-Train: step: 7800; ce: 3.610198; steps/sec: 11.69;
    FastEstimator-Train: step: 7820; epoch: 20; epoch_time: 33.09 sec;
    FastEstimator-Eval: step: 7820; epoch: 20; ce: 2.6623895; max_mcc: 0.37981499872517466; mcc: 0.37081626605247703; since_best_mcc: 1;
    FastEstimator-Train: step: 8100; ce: 3.5334125; steps/sec: 13.53;
    FastEstimator-Train: step: 8211; epoch: 21; epoch_time: 28.74 sec;
    FastEstimator-Eval: step: 8211; epoch: 21; ce: 2.6254454; max_mcc: 0.37981499872517466; mcc: 0.37584234647498965; since_best_mcc: 2;
    FastEstimator-Train: step: 8400; ce: 3.793233; steps/sec: 13.21;
    FastEstimator-Train: step: 8602; epoch: 22; epoch_time: 30.39 sec;
    FastEstimator-Eval: step: 8602; epoch: 22; ce: 2.6189961; max_mcc: 0.37981499872517466; mcc: 0.37361962631360524; since_best_mcc: 3;
    FastEstimator-Train: step: 8700; ce: 3.3799744; steps/sec: 12.41;
    FastEstimator-Train: step: 8993; epoch: 23; epoch_time: 33.65 sec;
    FastEstimator-Eval: step: 8993; epoch: 23; ce: 2.612086; max_mcc: 0.37981499872517466; mcc: 0.36652782752936447; since_best_mcc: 4;
    FastEstimator-Train: step: 9000; ce: 3.5184188; steps/sec: 11.46;
    FastEstimator-Train: step: 9300; ce: 3.8190534; steps/sec: 12.45;
    FastEstimator-Train: step: 9384; epoch: 24; epoch_time: 31.85 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 9384; epoch: 24; ce: 2.6104712; max_mcc: 0.3827068080998433; mcc: 0.3827068080998433; since_best_mcc: 0;
    FastEstimator-Train: step: 9600; ce: 3.4158635; steps/sec: 11.32;
    FastEstimator-Train: step: 9775; epoch: 25; epoch_time: 37.56 sec;
    FastEstimator-Eval: step: 9775; epoch: 25; ce: 2.6253555; max_mcc: 0.3827068080998433; mcc: 0.3780571077870386; since_best_mcc: 1;
    FastEstimator-Train: step: 9900; ce: 3.5892882; steps/sec: 10.56;
    FastEstimator-Train: step: 10166; epoch: 26; epoch_time: 32.24 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 10166; epoch: 26; ce: 2.570734; max_mcc: 0.3860799524151958; mcc: 0.3860799524151958; since_best_mcc: 0;
    FastEstimator-Train: step: 10200; ce: 3.202005; steps/sec: 12.08;
    FastEstimator-Train: step: 10500; ce: 3.6902063; steps/sec: 12.82;
    FastEstimator-Train: step: 10557; epoch: 27; epoch_time: 31.22 sec;
    FastEstimator-Eval: step: 10557; epoch: 27; ce: 2.596402; max_mcc: 0.3860799524151958; mcc: 0.3775887276776531; since_best_mcc: 1;
    FastEstimator-Train: step: 10800; ce: 3.4967391; steps/sec: 10.97;
    FastEstimator-Train: step: 10948; epoch: 28; epoch_time: 35.58 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 10948; epoch: 28; ce: 2.5560594; max_mcc: 0.39575154435306614; mcc: 0.39575154435306614; since_best_mcc: 0;
    FastEstimator-Train: step: 11100; ce: 3.3257341; steps/sec: 10.43;
    FastEstimator-Train: step: 11339; epoch: 29; epoch_time: 37.73 sec;
    FastEstimator-Eval: step: 11339; epoch: 29; ce: 2.590526; max_mcc: 0.39575154435306614; mcc: 0.3815712551874175; since_best_mcc: 1;
    FastEstimator-Train: step: 11400; ce: 3.620838; steps/sec: 10.83;
    FastEstimator-Train: step: 11700; ce: 3.507379; steps/sec: 10.39;
    FastEstimator-Train: step: 11730; epoch: 30; epoch_time: 38.13 sec;
    FastEstimator-Eval: step: 11730; epoch: 30; ce: 2.5580335; max_mcc: 0.39575154435306614; mcc: 0.3873913194436312; since_best_mcc: 2;
    FastEstimator-Train: step: 12000; ce: 3.4382076; steps/sec: 8.22;
    FastEstimator-Train: step: 12121; epoch: 31; epoch_time: 46.18 sec;
    FastEstimator-Eval: step: 12121; epoch: 31; ce: 2.5701263; max_mcc: 0.39575154435306614; mcc: 0.3885461189785508; since_best_mcc: 3;
    FastEstimator-Train: step: 12300; ce: 3.6822195; steps/sec: 8.56;
    FastEstimator-Train: step: 12512; epoch: 32; epoch_time: 42.19 sec;
    FastEstimator-Eval: step: 12512; epoch: 32; ce: 2.580277; max_mcc: 0.39575154435306614; mcc: 0.3831673757753597; since_best_mcc: 4;
    FastEstimator-Train: step: 12600; ce: 3.6753204; steps/sec: 10.64;
    FastEstimator-Train: step: 12900; ce: 3.3388412; steps/sec: 10.3;
    FastEstimator-Train: step: 12903; epoch: 33; epoch_time: 37.33 sec;
    FastEstimator-Eval: step: 12903; epoch: 33; ce: 2.5383658; max_mcc: 0.39575154435306614; mcc: 0.3899208164712791; since_best_mcc: 5;
    FastEstimator-Train: step: 13200; ce: 3.459231; steps/sec: 11.12;
    FastEstimator-Train: step: 13294; epoch: 34; epoch_time: 34.77 sec;
    FastEstimator-Eval: step: 13294; epoch: 34; ce: 2.5615156; max_mcc: 0.39575154435306614; mcc: 0.39374828935631234; since_best_mcc: 6;
    FastEstimator-Train: step: 13500; ce: 3.5448508; steps/sec: 11.49;
    FastEstimator-Train: step: 13685; epoch: 35; epoch_time: 33.58 sec;
    FastEstimator-Eval: step: 13685; epoch: 35; ce: 2.5683808; max_mcc: 0.39575154435306614; mcc: 0.3908550407055648; since_best_mcc: 7;
    FastEstimator-Train: step: 13800; ce: 3.4196963; steps/sec: 11.77;
    FastEstimator-Train: step: 14076; epoch: 36; epoch_time: 32.19 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 14076; epoch: 36; ce: 2.5312858; max_mcc: 0.39662035768091214; mcc: 0.39662035768091214; since_best_mcc: 0;
    FastEstimator-Train: step: 14100; ce: 3.5711946; steps/sec: 12.09;
    FastEstimator-Train: step: 14400; ce: 3.3513265; steps/sec: 12.34;
    FastEstimator-Train: step: 14467; epoch: 37; epoch_time: 32.73 sec;
    FastEstimator-Eval: step: 14467; epoch: 37; ce: 2.5464592; max_mcc: 0.39662035768091214; mcc: 0.3918940113625829; since_best_mcc: 1;
    FastEstimator-Train: step: 14700; ce: 3.428709; steps/sec: 10.19;
    FastEstimator-Train: step: 14858; epoch: 38; epoch_time: 39.47 sec;
    FastEstimator-Eval: step: 14858; epoch: 38; ce: 2.550658; max_mcc: 0.39662035768091214; mcc: 0.39033128081160223; since_best_mcc: 2;
    FastEstimator-Train: step: 15000; ce: 3.078777; steps/sec: 9.56;
    FastEstimator-Train: step: 15249; epoch: 39; epoch_time: 42.61 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 15249; epoch: 39; ce: 2.5513768; max_mcc: 0.39823307834973204; mcc: 0.39823307834973204; since_best_mcc: 0;
    FastEstimator-Train: step: 15300; ce: 3.3244073; steps/sec: 9.15;
    FastEstimator-Train: step: 15600; ce: 3.534006; steps/sec: 9.38;
    FastEstimator-Train: step: 15640; epoch: 40; epoch_time: 42.79 sec;
    FastEstimator-Eval: step: 15640; epoch: 40; ce: 2.5463102; max_mcc: 0.39823307834973204; mcc: 0.3910923978590251; since_best_mcc: 1;
    FastEstimator-Train: step: 15900; ce: 3.4411178; steps/sec: 8.62;
    FastEstimator-Train: step: 16031; epoch: 41; epoch_time: 43.14 sec;
    FastEstimator-Eval: step: 16031; epoch: 41; ce: 2.5373654; max_mcc: 0.39823307834973204; mcc: 0.3937565293374741; since_best_mcc: 2;
    FastEstimator-Train: step: 16200; ce: 3.544276; steps/sec: 9.21;
    FastEstimator-Train: step: 16422; epoch: 42; epoch_time: 40.7 sec;
    FastEstimator-Eval: step: 16422; epoch: 42; ce: 2.5232975; max_mcc: 0.39823307834973204; mcc: 0.3965550337054476; since_best_mcc: 3;
    FastEstimator-Train: step: 16500; ce: 3.2576585; steps/sec: 10.29;
    FastEstimator-Train: step: 16800; ce: 3.224753; steps/sec: 12.09;
    FastEstimator-Train: step: 16813; epoch: 43; epoch_time: 33.55 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Eval: step: 16813; epoch: 43; ce: 2.53742; max_mcc: 0.4034960526140117; mcc: 0.4034960526140117; since_best_mcc: 0;
    FastEstimator-Train: step: 17100; ce: 3.6015034; steps/sec: 10.15;
    FastEstimator-Train: step: 17204; epoch: 44; epoch_time: 38.7 sec;
    FastEstimator-Eval: step: 17204; epoch: 44; ce: 2.556477; max_mcc: 0.4034960526140117; mcc: 0.3878614516799426; since_best_mcc: 1;
    FastEstimator-Train: step: 17400; ce: 3.646784; steps/sec: 10.38;
    FastEstimator-Train: step: 17595; epoch: 45; epoch_time: 36.03 sec;
    FastEstimator-Eval: step: 17595; epoch: 45; ce: 2.535374; max_mcc: 0.4034960526140117; mcc: 0.3932205804029834; since_best_mcc: 2;
    FastEstimator-Train: step: 17700; ce: 3.4570975; steps/sec: 10.81;
    FastEstimator-Train: step: 17986; epoch: 46; epoch_time: 35.46 sec;
    FastEstimator-Eval: step: 17986; epoch: 46; ce: 2.5392685; max_mcc: 0.4034960526140117; mcc: 0.39483071780864193; since_best_mcc: 3;
    FastEstimator-Train: step: 18000; ce: 3.2406769; steps/sec: 11.2;
    FastEstimator-Train: step: 18300; ce: 3.3581827; steps/sec: 11.95;
    FastEstimator-Train: step: 18377; epoch: 47; epoch_time: 33.33 sec;
    FastEstimator-Eval: step: 18377; epoch: 47; ce: 2.5425549; max_mcc: 0.4034960526140117; mcc: 0.39340897654492785; since_best_mcc: 4;
    FastEstimator-Train: step: 18600; ce: 3.397046; steps/sec: 11.1;
    FastEstimator-Train: step: 18768; epoch: 48; epoch_time: 36.79 sec;
    FastEstimator-Eval: step: 18768; epoch: 48; ce: 2.536802; max_mcc: 0.4034960526140117; mcc: 0.3982097200874583; since_best_mcc: 5;
    FastEstimator-Train: step: 18900; ce: 3.3445437; steps/sec: 10.62;
    FastEstimator-Train: step: 19159; epoch: 49; epoch_time: 32.64 sec;
    FastEstimator-Eval: step: 19159; epoch: 49; ce: 2.5242054; max_mcc: 0.4034960526140117; mcc: 0.4025940945610127; since_best_mcc: 6;
    FastEstimator-Train: step: 19200; ce: 3.3015854; steps/sec: 12.12;
    FastEstimator-Train: step: 19500; ce: 3.3875098; steps/sec: 10.84;
    FastEstimator-Train: step: 19550; epoch: 50; epoch_time: 36.61 sec;
    FastEstimator-Eval: step: 19550; epoch: 50; ce: 2.5269408; max_mcc: 0.4034960526140117; mcc: 0.39490728915805207; since_best_mcc: 7;
    FastEstimator-BestModelSaver: Restoring model from /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model_best_mcc.h5
    FastEstimator-Finish: step: 19550; model_lr: 0.001; total_time: 1816.62 sec;


## Step 4 - Train a model using SuperLoss

Now it's time to try using SuperLoss to see whether curriculum learning can help us overcome our label noise. Note how easy it is to add SuperLoss to any existing loss function:


```python
loss = SuperLoss(CrossEntropy(inputs=("y_pred", "y"), outputs="ce"), output_confidence="confidence")  # The output_confidence arg is only needed if you want to visualize
estimator_super = build_estimator(loss)
superL = estimator_super.fit("SuperLoss")
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; logging_interval: 300; num_device: 0;
    FastEstimator-Train: step: 1; ce: -0.37075347;
    FastEstimator-Train: step: 300; ce: -0.8623735; steps/sec: 11.54;
    FastEstimator-Train: step: 391; epoch: 1; epoch_time: 37.39 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 391; epoch: 1; ce: -0.7983143; max_mcc: 0.1125597474520802; mcc: 0.1125597474520802; since_best_mcc: 0;
    FastEstimator-Train: step: 600; ce: -1.2613001; steps/sec: 10.75;
    FastEstimator-Train: step: 782; epoch: 2; epoch_time: 34.57 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 782; epoch: 2; ce: -1.3001225; max_mcc: 0.2365398720722943; mcc: 0.2365398720722943; since_best_mcc: 0;
    FastEstimator-Train: step: 900; ce: -1.3985932; steps/sec: 10.84;
    FastEstimator-Train: step: 1173; epoch: 3; epoch_time: 36.67 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 1173; epoch: 3; ce: -1.1731328; max_mcc: 0.26350025590618864; mcc: 0.26350025590618864; since_best_mcc: 0;
    FastEstimator-Train: step: 1200; ce: -1.949363; steps/sec: 10.66;
    FastEstimator-Train: step: 1500; ce: -1.1462929; steps/sec: 11.29;
    FastEstimator-Train: step: 1564; epoch: 4; epoch_time: 35.29 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 1564; epoch: 4; ce: -1.1312404; max_mcc: 0.30261112753359964; mcc: 0.30261112753359964; since_best_mcc: 0;
    FastEstimator-Train: step: 1800; ce: -1.2050605; steps/sec: 10.58;
    FastEstimator-Train: step: 1955; epoch: 5; epoch_time: 36.51 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 1955; epoch: 5; ce: -1.1527275; max_mcc: 0.3108380249377218; mcc: 0.3108380249377218; since_best_mcc: 0;
    FastEstimator-Train: step: 2100; ce: -1.9632031; steps/sec: 10.84;
    FastEstimator-Train: step: 2346; epoch: 6; epoch_time: 34.9 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 2346; epoch: 6; ce: -1.1548951; max_mcc: 0.3302925891657191; mcc: 0.3302925891657191; since_best_mcc: 0;
    FastEstimator-Train: step: 2400; ce: -1.3713049; steps/sec: 11.45;
    FastEstimator-Train: step: 2700; ce: -2.1784606; steps/sec: 12.54;
    FastEstimator-Train: step: 2737; epoch: 7; epoch_time: 31.73 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 2737; epoch: 7; ce: -1.1013749; max_mcc: 0.33298209578087556; mcc: 0.33298209578087556; since_best_mcc: 0;
    FastEstimator-Train: step: 3000; ce: -2.2637691; steps/sec: 11.29;
    FastEstimator-Train: step: 3128; epoch: 8; epoch_time: 34.81 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 3128; epoch: 8; ce: -1.1379492; max_mcc: 0.33374520314307604; mcc: 0.33374520314307604; since_best_mcc: 0;
    FastEstimator-Train: step: 3300; ce: -1.6673214; steps/sec: 11.56;
    FastEstimator-Train: step: 3519; epoch: 9; epoch_time: 32.83 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 3519; epoch: 9; ce: -1.1758204; max_mcc: 0.35506256211556864; mcc: 0.35506256211556864; since_best_mcc: 0;
    FastEstimator-Train: step: 3600; ce: -2.1572313; steps/sec: 11.72;
    FastEstimator-Train: step: 3900; ce: -2.438948; steps/sec: 11.93;
    FastEstimator-Train: step: 3910; epoch: 10; epoch_time: 33.52 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 3910; epoch: 10; ce: -1.0977045; max_mcc: 0.3612057969410237; mcc: 0.3612057969410237; since_best_mcc: 0;
    FastEstimator-Train: step: 4200; ce: -2.103427; steps/sec: 11.77;
    FastEstimator-Train: step: 4301; epoch: 11; epoch_time: 33.16 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 4301; epoch: 11; ce: -1.129376; max_mcc: 0.3688561000957651; mcc: 0.3688561000957651; since_best_mcc: 0;
    FastEstimator-Train: step: 4500; ce: -2.2184067; steps/sec: 11.33;
    FastEstimator-Train: step: 4692; epoch: 12; epoch_time: 34.78 sec;
    FastEstimator-Eval: step: 4692; epoch: 12; ce: -1.1229738; max_mcc: 0.3688561000957651; mcc: 0.3682739970648438; since_best_mcc: 1;
    FastEstimator-Train: step: 4800; ce: -1.8956504; steps/sec: 10.98;
    FastEstimator-Train: step: 5083; epoch: 13; epoch_time: 37.78 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 5083; epoch: 13; ce: -1.1398726; max_mcc: 0.38006988757755594; mcc: 0.38006988757755594; since_best_mcc: 0;
    FastEstimator-Train: step: 5100; ce: -1.612635; steps/sec: 10.2;
    FastEstimator-Train: step: 5400; ce: -2.6114616; steps/sec: 10.99;
    FastEstimator-Train: step: 5474; epoch: 14; epoch_time: 35.84 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 5474; epoch: 14; ce: -1.1187962; max_mcc: 0.3867649665900185; mcc: 0.3867649665900185; since_best_mcc: 0;
    FastEstimator-Train: step: 5700; ce: -1.9113309; steps/sec: 11.09;
    FastEstimator-Train: step: 5865; epoch: 15; epoch_time: 35.21 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 5865; epoch: 15; ce: -1.1155822; max_mcc: 0.3891158484813984; mcc: 0.3891158484813984; since_best_mcc: 0;
    FastEstimator-Train: step: 6000; ce: -2.168588; steps/sec: 11.55;
    FastEstimator-Train: step: 6256; epoch: 16; epoch_time: 32.14 sec;
    FastEstimator-Eval: step: 6256; epoch: 16; ce: -1.1226286; max_mcc: 0.3891158484813984; mcc: 0.38130873837010243; since_best_mcc: 1;
    FastEstimator-Train: step: 6300; ce: -2.4116144; steps/sec: 12.16;
    FastEstimator-Train: step: 6600; ce: -2.4472303; steps/sec: 12.48;
    FastEstimator-Train: step: 6647; epoch: 17; epoch_time: 31.88 sec;
    FastEstimator-Eval: step: 6647; epoch: 17; ce: -1.1562691; max_mcc: 0.3891158484813984; mcc: 0.38586837056326745; since_best_mcc: 2;
    FastEstimator-Train: step: 6900; ce: -2.1152298; steps/sec: 10.9;
    FastEstimator-Train: step: 7038; epoch: 18; epoch_time: 36.06 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 7038; epoch: 18; ce: -1.1300958; max_mcc: 0.39973141399120576; mcc: 0.39973141399120576; since_best_mcc: 0;
    FastEstimator-Train: step: 7200; ce: -2.357912; steps/sec: 11.42;
    FastEstimator-Train: step: 7429; epoch: 19; epoch_time: 32.38 sec;
    FastEstimator-Eval: step: 7429; epoch: 19; ce: -1.1756989; max_mcc: 0.39973141399120576; mcc: 0.3970311009909747; since_best_mcc: 1;
    FastEstimator-Train: step: 7500; ce: -1.9553542; steps/sec: 12.02;
    FastEstimator-Train: step: 7800; ce: -2.290727; steps/sec: 11.89;
    FastEstimator-Train: step: 7820; epoch: 20; epoch_time: 33.31 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 7820; epoch: 20; ce: -1.1249373; max_mcc: 0.40001165378893744; mcc: 0.40001165378893744; since_best_mcc: 0;
    FastEstimator-Train: step: 8100; ce: -1.96353; steps/sec: 12.03;
    FastEstimator-Train: step: 8211; epoch: 21; epoch_time: 32.26 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 8211; epoch: 21; ce: -1.1495345; max_mcc: 0.4173464588354924; mcc: 0.4173464588354924; since_best_mcc: 0;
    FastEstimator-Train: step: 8400; ce: -2.077148; steps/sec: 11.75;
    FastEstimator-Train: step: 8602; epoch: 22; epoch_time: 33.58 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 8602; epoch: 22; ce: -1.1274261; max_mcc: 0.4224403126729616; mcc: 0.4224403126729616; since_best_mcc: 0;
    FastEstimator-Train: step: 8700; ce: -1.6582375; steps/sec: 11.61;
    FastEstimator-Train: step: 8993; epoch: 23; epoch_time: 33.55 sec;
    FastEstimator-Eval: step: 8993; epoch: 23; ce: -1.1531212; max_mcc: 0.4224403126729616; mcc: 0.41700806610446517; since_best_mcc: 1;
    FastEstimator-Train: step: 9000; ce: -2.614518; steps/sec: 11.64;
    FastEstimator-Train: step: 9300; ce: -2.4157054; steps/sec: 11.93;
    FastEstimator-Train: step: 9384; epoch: 24; epoch_time: 33.17 sec;
    FastEstimator-Eval: step: 9384; epoch: 24; ce: -1.1839094; max_mcc: 0.4224403126729616; mcc: 0.3992353404215677; since_best_mcc: 2;
    FastEstimator-Train: step: 9600; ce: -2.136599; steps/sec: 12.04;
    FastEstimator-Train: step: 9775; epoch: 25; epoch_time: 31.31 sec;
    FastEstimator-Eval: step: 9775; epoch: 25; ce: -1.1641603; max_mcc: 0.4224403126729616; mcc: 0.42078412402594256; since_best_mcc: 3;
    FastEstimator-Train: step: 9900; ce: -1.9771264; steps/sec: 12.72;
    FastEstimator-Train: step: 10166; epoch: 26; epoch_time: 30.82 sec;
    FastEstimator-Eval: step: 10166; epoch: 26; ce: -1.1250267; max_mcc: 0.4224403126729616; mcc: 0.4181807901838999; since_best_mcc: 4;
    FastEstimator-Train: step: 10200; ce: -2.4677532; steps/sec: 12.61;
    FastEstimator-Train: step: 10500; ce: -2.3928475; steps/sec: 12.42;
    FastEstimator-Train: step: 10557; epoch: 27; epoch_time: 31.78 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 10557; epoch: 27; ce: -1.1390682; max_mcc: 0.43011411759582024; mcc: 0.43011411759582024; since_best_mcc: 0;
    FastEstimator-Train: step: 10800; ce: -2.350306; steps/sec: 12.28;
    FastEstimator-Train: step: 10948; epoch: 28; epoch_time: 31.89 sec;
    FastEstimator-Eval: step: 10948; epoch: 28; ce: -1.1232755; max_mcc: 0.43011411759582024; mcc: 0.42548395606582284; since_best_mcc: 1;
    FastEstimator-Train: step: 11100; ce: -2.4872317; steps/sec: 12.25;
    FastEstimator-Train: step: 11339; epoch: 29; epoch_time: 30.53 sec;
    FastEstimator-Eval: step: 11339; epoch: 29; ce: -1.1274782; max_mcc: 0.43011411759582024; mcc: 0.4255082704606608; since_best_mcc: 2;
    FastEstimator-Train: step: 11400; ce: -2.769877; steps/sec: 12.93;
    FastEstimator-Train: step: 11700; ce: -1.4690123; steps/sec: 12.88;
    FastEstimator-Train: step: 11730; epoch: 30; epoch_time: 30.88 sec;
    FastEstimator-Eval: step: 11730; epoch: 30; ce: -1.1748626; max_mcc: 0.43011411759582024; mcc: 0.4123991247445033; since_best_mcc: 3;
    FastEstimator-Train: step: 12000; ce: -2.3541317; steps/sec: 13.03;
    FastEstimator-Train: step: 12121; epoch: 31; epoch_time: 30.0 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 12121; epoch: 31; ce: -1.1842433; max_mcc: 0.43048319026023396; mcc: 0.43048319026023396; since_best_mcc: 0;
    FastEstimator-Train: step: 12300; ce: -2.3778152; steps/sec: 12.49;
    FastEstimator-Train: step: 12512; epoch: 32; epoch_time: 33.83 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 12512; epoch: 32; ce: -1.1425531; max_mcc: 0.43271764482253255; mcc: 0.43271764482253255; since_best_mcc: 0;
    FastEstimator-Train: step: 12600; ce: -1.6453986; steps/sec: 11.02;
    FastEstimator-Train: step: 12900; ce: -2.376278; steps/sec: 11.84;
    FastEstimator-Train: step: 12903; epoch: 33; epoch_time: 33.78 sec;
    FastEstimator-Eval: step: 12903; epoch: 33; ce: -1.128669; max_mcc: 0.43271764482253255; mcc: 0.4238694483648559; since_best_mcc: 1;
    FastEstimator-Train: step: 13200; ce: -2.389115; steps/sec: 11.7;
    FastEstimator-Train: step: 13294; epoch: 34; epoch_time: 33.13 sec;
    FastEstimator-Eval: step: 13294; epoch: 34; ce: -1.1393003; max_mcc: 0.43271764482253255; mcc: 0.43174883457168434; since_best_mcc: 2;
    FastEstimator-Train: step: 13500; ce: -1.987546; steps/sec: 11.99;
    FastEstimator-Train: step: 13685; epoch: 35; epoch_time: 32.03 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 13685; epoch: 35; ce: -1.1264135; max_mcc: 0.4343215901949135; mcc: 0.4343215901949135; since_best_mcc: 0;
    FastEstimator-Train: step: 13800; ce: -1.5252323; steps/sec: 12.19;
    FastEstimator-Train: step: 14076; epoch: 36; epoch_time: 33.29 sec;
    FastEstimator-Eval: step: 14076; epoch: 36; ce: -1.1500013; max_mcc: 0.4343215901949135; mcc: 0.42942850564856067; since_best_mcc: 1;
    FastEstimator-Train: step: 14100; ce: -2.5956576; steps/sec: 11.38;
    FastEstimator-Train: step: 14400; ce: -2.2249644; steps/sec: 11.29;
    FastEstimator-Train: step: 14467; epoch: 37; epoch_time: 34.99 sec;
    FastEstimator-Eval: step: 14467; epoch: 37; ce: -1.1593236; max_mcc: 0.4343215901949135; mcc: 0.42860970989849045; since_best_mcc: 2;
    FastEstimator-Train: step: 14700; ce: -2.3185768; steps/sec: 10.72;
    FastEstimator-Train: step: 14858; epoch: 38; epoch_time: 36.61 sec;
    FastEstimator-Eval: step: 14858; epoch: 38; ce: -1.2385705; max_mcc: 0.4343215901949135; mcc: 0.4341883370113877; since_best_mcc: 3;
    FastEstimator-Train: step: 15000; ce: -2.1548734; steps/sec: 11.19;
    FastEstimator-Train: step: 15249; epoch: 39; epoch_time: 34.31 sec;
    FastEstimator-Eval: step: 15249; epoch: 39; ce: -1.1571573; max_mcc: 0.4343215901949135; mcc: 0.42741634385167887; since_best_mcc: 4;
    FastEstimator-Train: step: 15300; ce: -2.2904358; steps/sec: 11.42;
    FastEstimator-Train: step: 15600; ce: -2.377496; steps/sec: 12.27;
    FastEstimator-Train: step: 15640; epoch: 40; epoch_time: 32.42 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 15640; epoch: 40; ce: -1.1836212; max_mcc: 0.4365484048719312; mcc: 0.4365484048719312; since_best_mcc: 0;
    FastEstimator-Train: step: 15900; ce: -2.0115094; steps/sec: 12.36;
    FastEstimator-Train: step: 16031; epoch: 41; epoch_time: 30.95 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 16031; epoch: 41; ce: -1.1716866; max_mcc: 0.4405922201097312; mcc: 0.4405922201097312; since_best_mcc: 0;
    FastEstimator-Train: step: 16200; ce: -2.5693574; steps/sec: 12.73;
    FastEstimator-Train: step: 16422; epoch: 42; epoch_time: 31.04 sec;
    FastEstimator-Eval: step: 16422; epoch: 42; ce: -1.187574; max_mcc: 0.4405922201097312; mcc: 0.438273179007191; since_best_mcc: 1;
    FastEstimator-Train: step: 16500; ce: -2.7689743; steps/sec: 12.4;
    FastEstimator-Train: step: 16800; ce: -2.157916; steps/sec: 12.79;
    FastEstimator-Train: step: 16813; epoch: 43; epoch_time: 31.19 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 16813; epoch: 43; ce: -1.1804336; max_mcc: 0.44704770369541796; mcc: 0.44704770369541796; since_best_mcc: 0;
    FastEstimator-Train: step: 17100; ce: -2.44828; steps/sec: 11.87;
    FastEstimator-Train: step: 17204; epoch: 44; epoch_time: 33.85 sec;
    FastEstimator-Eval: step: 17204; epoch: 44; ce: -1.1787541; max_mcc: 0.44704770369541796; mcc: 0.4378045792705017; since_best_mcc: 1;
    FastEstimator-Train: step: 17400; ce: -1.9475007; steps/sec: 11.25;
    FastEstimator-Train: step: 17595; epoch: 45; epoch_time: 32.59 sec;
    FastEstimator-Eval: step: 17595; epoch: 45; ce: -1.1700599; max_mcc: 0.44704770369541796; mcc: 0.4341889164769888; since_best_mcc: 2;
    FastEstimator-Train: step: 17700; ce: -2.5675113; steps/sec: 12.09;
    FastEstimator-Train: step: 17986; epoch: 46; epoch_time: 34.46 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Eval: step: 17986; epoch: 46; ce: -1.1674488; max_mcc: 0.4508618065428228; mcc: 0.4508618065428228; since_best_mcc: 0;
    FastEstimator-Train: step: 18000; ce: -2.265307; steps/sec: 11.1;
    FastEstimator-Train: step: 18300; ce: -2.2236004; steps/sec: 10.81;
    FastEstimator-Train: step: 18377; epoch: 47; epoch_time: 36.58 sec;
    FastEstimator-Eval: step: 18377; epoch: 47; ce: -1.1883423; max_mcc: 0.4508618065428228; mcc: 0.43941256813027796; since_best_mcc: 1;
    FastEstimator-Train: step: 18600; ce: -2.2972271; steps/sec: 11.0;
    FastEstimator-Train: step: 18768; epoch: 48; epoch_time: 36.03 sec;
    FastEstimator-Eval: step: 18768; epoch: 48; ce: -1.1837901; max_mcc: 0.4508618065428228; mcc: 0.4390055473254998; since_best_mcc: 2;
    FastEstimator-Train: step: 18900; ce: -2.659154; steps/sec: 10.33;
    FastEstimator-Train: step: 19159; epoch: 49; epoch_time: 36.53 sec;
    FastEstimator-Eval: step: 19159; epoch: 49; ce: -1.2066782; max_mcc: 0.4508618065428228; mcc: 0.4438185517620055; since_best_mcc: 3;
    FastEstimator-Train: step: 19200; ce: -1.8844283; steps/sec: 10.95;
    FastEstimator-Train: step: 19500; ce: -3.425774; steps/sec: 11.65;
    FastEstimator-Train: step: 19550; epoch: 50; epoch_time: 34.0 sec;
    FastEstimator-Eval: step: 19550; epoch: 50; ce: -1.2063822; max_mcc: 0.4508618065428228; mcc: 0.4422894428621687; since_best_mcc: 4;
    FastEstimator-BestModelSaver: Restoring model from /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8r026lm8/model1_best_mcc.h5
    FastEstimator-Finish: step: 19550; model1_lr: 0.001; total_time: 1788.57 sec;


## Step 5 - Performance Comparison

Let's take a look at how each of the final models compare:


```python
estimator_regular.test()
```

    FastEstimator-Test: step: 19550; epoch: 50; ce: 2.54847; mcc: 0.40158894368465714;





    <fastestimator.summary.summary.Summary at 0x17da64160>




```python
estimator_super.test()
```

    FastEstimator-Test: step: 19550; epoch: 50; ce: -1.2840478; mcc: 0.4381719790823748;





    <fastestimator.summary.summary.Summary at 0x17e7239b0>




```python
fe.summary.logs.visualize_logs([regular, superL], include_metrics={'mcc', 'ce', 'max_mcc'})
```


    
![png](assets/branches/r1.2/example/curriculum_learning/superloss_files/superloss_15_0.png)
    


As we can see from the results above, a simple 1 line change to add SuperLoss into the training procedure can raise our model's mcc by a full 4 or 5 points in the presence of noisy input labels. Let's also take a look at the confidence scores generated by SuperLoss on the noisy vs clean data:


```python
fe.summary.logs.visualize_logs(estimator_super.system.custom_graphs['label_confidence'])
```


    
![png](assets/branches/r1.2/example/curriculum_learning/superloss_files/superloss_17_0.png)
    


As the graph above demonstrates, the corrupted samples have significantly lower average confidence scores than the clean samples. This is also true when we analyze the confidence scores during regular training, but the separation is not as strong:


```python
fe.summary.logs.visualize_logs(estimator_regular.system.custom_graphs['label_confidence'])
```


    
![png](assets/branches/r1.2/example/curriculum_learning/superloss_files/superloss_19_0.png)
    

