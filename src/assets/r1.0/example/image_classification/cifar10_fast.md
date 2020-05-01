# Cifar-10 Image Classification Example Using ResNet (Pytorch Backend)
In this example we are going to demonstrate how to train a Cifar-10 image classification model using ResNet architecture with Pytorch backend. All training details including model structure, data preprocessing, learning rate control ... etc comes from the reference of https://github.com/davidcpage/cifar10-fast.

## Import the required libraries


```python
import fastestimator as fe
import numpy as np
import matplotlib.pyplot as plt
import tempfile
```


```python
#training parameters
epochs = 24
batch_size = 512
max_steps_per_epoch = None
save_dir = tempfile.mkdtemp()
```

## Step 1 - Data and `Pipeline` preparation
In this step, we will load Cifar-10 training and validation datasets and prepare FastEstimator's pipeline.

### Load dataset 
We use fastestimator API to load the Cifar-10 dataset and get the test set by splitting 50% evaluation set. 


```python
from fastestimator.dataset.data import cifar10

train_data, eval_data = cifar10.load_data()
test_data = eval_data.split(0.5)
```

### Set up preprocess pipeline
Here we start to set up the data pipeline which in this case needs data augmentation including randomly cropping, horizontal flipping, image obscuration and one-hot encoding for label. Beside the image channel need to be transpose to CHW due to Pytorch convention. We set up those processing step using `Ops` and meanwhile define the data source (loaded dataset) and batch size. 


```python
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize, Onehot
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop

pipeline = fe.Pipeline(
    train_data=train_data,
    eval_data=eval_data,
    test_data=test_data,
    batch_size=batch_size,
    ops=[
        Normalize(inputs="x", outputs="x_out", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        PadIfNeeded(min_height=40, min_width=40, image_in="x_out", image_out="x_out", mode="train"),
        RandomCrop(32, 32, image_in="x_out", image_out="x_out", mode="train"),
        Sometimes(HorizontalFlip(image_in="x_out", image_out="x_out", mode="train")),
        CoarseDropout(inputs="x_out", outputs="x_out", mode="train", max_holes=1),
        ChannelTranspose(inputs="x_out", outputs="x_out"),
        Onehot(inputs="y", outputs="y_out", mode="train", num_classes=10, label_smoothing=0.2)
    ])
```

### Validate `Pipeline`
In order to make sure the pipeline works as expected, we need to visualize the output of pipeline image and check its size.
`Pipeline.get_results` will return a batch data of pipeline output.


```python
data = pipeline.get_results()
data_xin = data["x"]
data_xout = data["x_out"]
data_yin = data["y"]
data_yout = data["y_out"]

print("the pipeline input image size: {}".format(data_xin.numpy().shape))
print("the pipeline output image size: {}".format(data_xout.numpy().shape))
print("the pipeline input label size: {}".format(data_yin.numpy().shape))
print("the pipeline output label size: {}".format(data_yout.numpy().shape))
```

    the pipeline input image size: (512, 32, 32, 3)
    the pipeline output image size: (512, 3, 32, 32)
    the pipeline input label size: (512, 1)
    the pipeline output label size: (512, 10)



```python
sample_num = 5

fig, axs = plt.subplots(sample_num, 2, figsize=(12,12))

axs[0,0].set_title("pipeline input img")
axs[0,1].set_title("pipeline output img")

for i, j in enumerate(np.random.randint(low=0, high=batch_size-1, size=sample_num)):
    # pipeline image visualization 
    img_in = data_xin.numpy()[j]
    axs[i,0].imshow(img_in)
    
    img_out = data_xout.numpy()[j].transpose((1,2,0))
    img_out[:,:,0] = img_out[:,:,0] * 0.2471 + 0.4914 
    img_out[:,:,1] = img_out[:,:,1] * 0.2435 + 0.4822
    img_out[:,:,2] = img_out[:,:,2] * 0.2616 + 0.4465
    axs[i,1].imshow(img_out)
    
    # pipeline label print
    label_in = data_yin.numpy()[j]
    label_out = data_yout.numpy()[j]
    print("label_in:{} -> label_out:{}".format(label_in, label_out))
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    label_in:[0] -> label_out:[0.82 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02]
    label_in:[4] -> label_out:[0.02 0.02 0.02 0.02 0.82 0.02 0.02 0.02 0.02 0.02]
    label_in:[9] -> label_out:[0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.82]
    label_in:[1] -> label_out:[0.02 0.82 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02]
    label_in:[0] -> label_out:[0.82 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02]



![png](assets/example/image_classification/cifar10_fast_files/cifar10_fast_10_2.png)


## Step 2 - `Network` construction
**FastEstimator supports both Pytorch and Tensorflow, so this section can use both backend to implement.** <br>
We are going to only demonstate the Pytorch way in this example.

### Model construction
The model definitions are implemented in Pytorch and instantiated by calling `fe.build` which also associates the model with specific optimizer.


```python
import torch
import torch.nn as nn
import torch.nn.functional as fn

class FastCifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, padding=(1, 1))
        self.conv0_bn = nn.BatchNorm2d(64, momentum=0.8)
        self.conv1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(128, momentum=0.8)
        self.residual1 = Residual(128, 128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(256, momentum=0.8)
        self.residual2 = Residual(256, 256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(512, momentum=0.8)
        self.residual3 = Residual(512, 512)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        # prep layer
        x = self.conv0(x)
        x = self.conv0_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        # layer 1
        x = self.conv1(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual1(x)
        # layer 2
        x = self.conv2(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual2(x)
        # layer 3
        x = self.conv3(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv3_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual3(x)
        # layer 4
        x = fn.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = fn.softmax(x, dim=-1)
        return x


class Residual(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(channel_out)
        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        return x

model = fe.build(model_fn=FastCifar, optimizer_fn="adam")
```

### `Network` defintion
`Ops` are the basic components of a network that include models, loss calculation units, posprocessing units. In this step we are going to combine those pieces togethers into `Network`   


```python
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp

network = fe.Network(ops=[
        ModelOp(model=model, inputs="x_out", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y_out"), outputs="ce", mode="train"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", mode="test"),
        UpdateOp(model=model, loss_name="ce", mode="train")
    ])
```

## Step 3 - `Estimator` definition and training
In this step, we define the `Estimator` to connect the `Network` with `Pipeline` and set the `traces` which will compute accuracy (Accuracy), save best model (BestModelSaver), and change learning rate (LRScheduler). At the end, we use `Estimator.fit` to trigger the training.


```python
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy

def lr_schedule(step):
    if step <= 490:
        lr = step / 490 * 0.4
    else:
        lr = (2352 - step) / 1862 * 0.4
    return lr * 0.1

traces = [
    Accuracy(true_key="y", pred_key="y_pred"),
    BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max"),
    LRScheduler(model=model, lr_fn=lr_schedule)
]

estimator = fe.Estimator(pipeline=pipeline,
                         network=network,
                         epochs=epochs,
                         traces=traces,
                         max_steps_per_epoch=max_steps_per_epoch)

estimator.fit() # start the training 
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; model_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 3.219695; model_lr: 8.163265e-05; 
    FastEstimator-Train: step: 98; epoch: 1; epoch_time: 9.92 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 98; epoch: 1; accuracy: 0.5402; 
    FastEstimator-Train: step: 100; ce: 1.5576406; steps/sec: 10.01; model_lr: 0.008163265; 
    FastEstimator-Train: step: 196; epoch: 2; epoch_time: 9.85 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 196; epoch: 2; accuracy: 0.6428; 
    FastEstimator-Train: step: 200; ce: 1.4646513; steps/sec: 9.99; model_lr: 0.01632653; 
    FastEstimator-Train: step: 294; epoch: 3; epoch_time: 9.82 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 294; epoch: 3; accuracy: 0.7756; 
    FastEstimator-Train: step: 300; ce: 1.3107454; steps/sec: 9.98; model_lr: 0.024489796; 
    FastEstimator-Train: step: 392; epoch: 4; epoch_time: 9.86 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 392; epoch: 4; accuracy: 0.7882; 
    FastEstimator-Train: step: 400; ce: 1.2392472; steps/sec: 9.95; model_lr: 0.03265306; 
    FastEstimator-Train: step: 490; epoch: 5; epoch_time: 9.86 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 490; epoch: 5; accuracy: 0.8258; 
    FastEstimator-Train: step: 500; ce: 1.1978259; steps/sec: 9.9; model_lr: 0.039785177; 
    FastEstimator-Train: step: 588; epoch: 6; epoch_time: 9.92 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 588; epoch: 6; accuracy: 0.8466; 
    FastEstimator-Train: step: 600; ce: 1.1920979; steps/sec: 9.94; model_lr: 0.03763695; 
    FastEstimator-Train: step: 686; epoch: 7; epoch_time: 9.88 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 686; epoch: 7; accuracy: 0.8686; 
    FastEstimator-Train: step: 700; ce: 1.1124842; steps/sec: 9.92; model_lr: 0.03548872; 
    FastEstimator-Train: step: 784; epoch: 8; epoch_time: 9.92 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 784; epoch: 8; accuracy: 0.8698; 
    FastEstimator-Train: step: 800; ce: 1.0877913; steps/sec: 9.91; model_lr: 0.033340495; 
    FastEstimator-Train: step: 882; epoch: 9; epoch_time: 9.9 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 882; epoch: 9; accuracy: 0.885; 
    FastEstimator-Train: step: 900; ce: 1.1029329; steps/sec: 9.9; model_lr: 0.031192265; 
    FastEstimator-Train: step: 980; epoch: 10; epoch_time: 9.91 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 980; epoch: 10; accuracy: 0.8972; 
    FastEstimator-Train: step: 1000; ce: 1.080683; steps/sec: 9.91; model_lr: 0.02904404; 
    FastEstimator-Train: step: 1078; epoch: 11; epoch_time: 9.94 sec; 
    FastEstimator-Eval: step: 1078; epoch: 11; accuracy: 0.895; 
    FastEstimator-Train: step: 1100; ce: 1.0479429; steps/sec: 9.89; model_lr: 0.026895812; 
    FastEstimator-Train: step: 1176; epoch: 12; epoch_time: 9.91 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 1176; epoch: 12; accuracy: 0.899; 
    FastEstimator-Train: step: 1200; ce: 1.0285817; steps/sec: 9.9; model_lr: 0.024747584; 
    FastEstimator-Train: step: 1274; epoch: 13; epoch_time: 9.89 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 1274; epoch: 13; accuracy: 0.901; 
    FastEstimator-Train: step: 1300; ce: 0.99329174; steps/sec: 9.87; model_lr: 0.022599356; 
    FastEstimator-Train: step: 1372; epoch: 14; epoch_time: 9.98 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 1372; epoch: 14; accuracy: 0.9116; 
    FastEstimator-Train: step: 1400; ce: 0.9965744; steps/sec: 9.85; model_lr: 0.020451128; 
    FastEstimator-Train: step: 1470; epoch: 15; epoch_time: 9.97 sec; 
    FastEstimator-Eval: step: 1470; epoch: 15; accuracy: 0.911; 
    FastEstimator-Train: step: 1500; ce: 0.9886917; steps/sec: 9.86; model_lr: 0.0183029; 
    FastEstimator-Train: step: 1568; epoch: 16; epoch_time: 9.96 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 1568; epoch: 16; accuracy: 0.9168; 
    FastEstimator-Train: step: 1600; ce: 0.98263824; steps/sec: 9.87; model_lr: 0.016154673; 
    FastEstimator-Train: step: 1666; epoch: 17; epoch_time: 9.94 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 1666; epoch: 17; accuracy: 0.924; 
    FastEstimator-Train: step: 1700; ce: 0.95481974; steps/sec: 9.88; model_lr: 0.014006444; 
    FastEstimator-Train: step: 1764; epoch: 18; epoch_time: 9.92 sec; 
    FastEstimator-Eval: step: 1764; epoch: 18; accuracy: 0.9214; 
    FastEstimator-Train: step: 1800; ce: 0.9707108; steps/sec: 9.88; model_lr: 0.011858217; 
    FastEstimator-Train: step: 1862; epoch: 19; epoch_time: 9.95 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 1862; epoch: 19; accuracy: 0.9288; 
    FastEstimator-Train: step: 1900; ce: 0.9509058; steps/sec: 9.86; model_lr: 0.00970999; 
    FastEstimator-Train: step: 1960; epoch: 20; epoch_time: 9.97 sec; 
    FastEstimator-Eval: step: 1960; epoch: 20; accuracy: 0.924; 
    FastEstimator-Train: step: 2000; ce: 0.9307832; steps/sec: 9.84; model_lr: 0.0075617614; 
    FastEstimator-Train: step: 2058; epoch: 21; epoch_time: 9.93 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 2058; epoch: 21; accuracy: 0.9334; 
    FastEstimator-Train: step: 2100; ce: 0.91421276; steps/sec: 9.84; model_lr: 0.0054135337; 
    FastEstimator-Train: step: 2156; epoch: 22; epoch_time: 10.0 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 2156; epoch: 22; accuracy: 0.9374; 
    FastEstimator-Train: step: 2200; ce: 0.9126489; steps/sec: 9.82; model_lr: 0.0032653061; 
    FastEstimator-Train: step: 2254; epoch: 23; epoch_time: 9.98 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 2254; epoch: 23; accuracy: 0.9412; 
    FastEstimator-Train: step: 2300; ce: 0.909637; steps/sec: 9.87; model_lr: 0.0011170784; 
    FastEstimator-Train: step: 2352; epoch: 24; epoch_time: 9.96 sec; 
    Saved model to /tmp/tmp8mcc0a80/model_best_accuracy.pt
    FastEstimator-Eval: step: 2352; epoch: 24; accuracy: 0.9418; 
    FastEstimator-Finish: step: 2352; total_time: 257.44 sec; model_lr: 0.0; 


## Model testing
`Estimator.test` will trigger model testing and runs model on test data that defined in `Pipeline`. Here we only setup the accuracy as evaluate


```python
estimator.test()
```

    FastEstimator-Test: epoch: 24; accuracy: 0.9362; 


## Images inference
In this step we run image inference directly using the model that just trained. We randomly select 5 images from testing dataset and infer them image by image with `Pipeline.transform` and `Netowork.transform`. Please be aware that the pipeline is no longer the same as it did in training, because we don't want to have data augmentation during inference. This detail was already defined in the `Pipeline` (mode = "!infer"). 


```python
sample_num = 5

fig, axs = plt.subplots(sample_num, 3, figsize=(12,12))

axs[0,0].set_title("pipeline input")
axs[0,1].set_title("pipeline output")
axs[0,2].set_title("predict result")

for i, j in enumerate(np.random.randint(low=0, high=batch_size-1, size=sample_num)):
    data = {"x": test_data["x"][j]}
    axs[i,0].imshow(data["x"], cmap="gray")
    
    # run the pipeline
    data = pipeline.transform(data, mode="infer") 
    img = data["x_out"].squeeze(axis=0).transpose((1,2,0))
    img[:,:,0] = img[:,:,0] * 0.2471 + 0.4914 
    img[:,:,1] = img[:,:,1] * 0.2435 + 0.4822
    img[:,:,2] = img[:,:,2] * 0.2616 + 0.4465
    axs[i,1].imshow(img)
    
    # run the network
    data = network.transform(data, mode="infer")
    predict = data["y_pred"].numpy().squeeze(axis=(0))
    axs[i,2].text(0.2, 0.5, "predicted class: {}".format(np.argmax(predict)))
    axs[i,2].get_xaxis().set_visible(False)
    axs[i,2].get_yaxis().set_visible(False)
```


![png](assets/example/image_classification/cifar10_fast_files/cifar10_fast_20_0.png)

