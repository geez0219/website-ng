# Adversarial Discriminative Domain Adaptation (ADDA) in FastEstimator

In this notebook we will demonstrate how to perform domain adaptation in FastEstimator.
Specifically, we will demonstrate one of adversarial training based domain adaptation methods, [*Adversarial Discriminative Domain Adaptation (ADDA)*](https://arxiv.org/abs/1702.05464). 

We will look at how to adapt a digit classifier trained on MNIST dataset to another digit dataset, USPS dataset.
The digit classifer is composed of a feature extractor network and a classifier.


```python
import os
import time

import tensorflow as tf
import numpy as np

import fastestimator as fe
```

## Input Data Pipeline
We will first download the two datasets using our dataset api.


```python
from fastestimator.dataset import mnist, usps
from fastestimator.op.numpyop import ImageReader
from fastestimator import RecordWriter
usps_train_csv, usps_eval_csv, usps_parent_dir = usps.load_data()
mnist_train_csv, mnist_eval_csv, mnist_parent_dir = mnist.load_data()
BATCH_SIZE = 128
```

    Extracting /root/fastestimator_data/USPS/zip.train.gz
    Extracting /root/fastestimator_data/USPS/zip.test.gz


The dataset api creates a train csv file with each row containing a relative path to a image and the class label.
Two train csv files will have the same column names.
We need to change these column names to unique name for our purpose.


```python
import pandas as pd
df = pd.read_csv(mnist_train_csv)
df.columns = ['source_img', 'source_label']
df.to_csv(mnist_train_csv, index=False)

df = pd.read_csv(usps_train_csv)
df.columns = ['target_img', 'target_label']
df.to_csv(usps_train_csv, index=False)
```

With the modified csv files, we can now create an input data pipeline that returns a batch from the MNIST dataset and the USPS dataset. 
#### Note that the input data pipeline created here is an unpaired dataset of the MNIST and the USPS.


```python
from fastestimator.op.tensorop import Resize, Minmax

writer = RecordWriter(save_dir=os.path.join(os.path.dirname(mnist_parent_dir), 'adda', 'tfr'),
                      train_data=(usps_train_csv, mnist_train_csv),
                      ops=(
                          [ImageReader(inputs="target_img", outputs="target_img", parent_path=usps_parent_dir, grey_scale=True)], # first tuple element
                          [ImageReader(inputs="source_img", outputs="source_img", parent_path=mnist_parent_dir, grey_scale=True)])) # second tuple element

```

We apply the following preprocessing to both datasets:
* Resize of images to $32\times32$
* Minmax pixel value normalization


```python
pipeline = fe.Pipeline(
    batch_size=BATCH_SIZE,
    data=writer,
    ops=[
        Resize(inputs="target_img", outputs="target_img", size=(32, 32)),
        Resize(inputs="source_img", outputs="source_img", size=(32, 32)),
        Minmax(inputs="target_img", outputs="target_img"),
        Minmax(inputs="source_img", outputs="source_img")
    ]
)
a = pipeline.show_results()
```

    FastEstimator: Saving tfrecord to /root/fastestimator_data/adda/tfr
    FastEstimator: Converting Train TFRecords 0.0%, Speed: 0.00 record/sec
    FastEstimator: Converting Train TFRecords 5.0%, Speed: 6438.19 record/sec
    FastEstimator: Converting Train TFRecords 10.0%, Speed: 9156.87 record/sec
    FastEstimator: Converting Train TFRecords 15.0%, Speed: 10721.43 record/sec
    FastEstimator: Converting Train TFRecords 20.0%, Speed: 10055.73 record/sec
    FastEstimator: Converting Train TFRecords 25.0%, Speed: 8862.50 record/sec
    FastEstimator: Converting Train TFRecords 30.0%, Speed: 9326.50 record/sec
    FastEstimator: Converting Train TFRecords 34.9%, Speed: 8585.62 record/sec
    FastEstimator: Converting Train TFRecords 39.9%, Speed: 8430.13 record/sec
    FastEstimator: Converting Train TFRecords 44.9%, Speed: 8324.00 record/sec
    FastEstimator: Converting Train TFRecords 49.9%, Speed: 8462.68 record/sec
    FastEstimator: Converting Train TFRecords 54.9%, Speed: 8178.13 record/sec
    FastEstimator: Converting Train TFRecords 59.9%, Speed: 8283.31 record/sec
    FastEstimator: Converting Train TFRecords 64.9%, Speed: 8706.65 record/sec
    FastEstimator: Converting Train TFRecords 69.9%, Speed: 9099.43 record/sec
    FastEstimator: Converting Train TFRecords 74.9%, Speed: 9475.19 record/sec
    FastEstimator: Converting Train TFRecords 79.9%, Speed: 9820.01 record/sec
    FastEstimator: Converting Train TFRecords 84.9%, Speed: 10149.50 record/sec
    FastEstimator: Converting Train TFRecords 89.9%, Speed: 10459.54 record/sec
    FastEstimator: Converting Train TFRecords 94.8%, Speed: 10757.69 record/sec
    FastEstimator: Converting Train TFRecords 99.8%, Speed: 11044.58 record/sec
    FastEstimator: Converting Train TFRecords 0.0%, Speed: 0.00 record/sec
    FastEstimator: Converting Train TFRecords 5.0%, Speed: 11054.88 record/sec
    FastEstimator: Converting Train TFRecords 10.0%, Speed: 12287.17 record/sec
    FastEstimator: Converting Train TFRecords 15.0%, Speed: 10449.15 record/sec
    FastEstimator: Converting Train TFRecords 20.0%, Speed: 9954.79 record/sec
    FastEstimator: Converting Train TFRecords 25.0%, Speed: 10100.48 record/sec
    FastEstimator: Converting Train TFRecords 30.0%, Speed: 9998.55 record/sec
    FastEstimator: Converting Train TFRecords 35.0%, Speed: 10080.56 record/sec
    FastEstimator: Converting Train TFRecords 40.0%, Speed: 10618.81 record/sec
    FastEstimator: Converting Train TFRecords 45.0%, Speed: 10902.46 record/sec
    FastEstimator: Converting Train TFRecords 50.0%, Speed: 11026.90 record/sec
    FastEstimator: Converting Train TFRecords 55.0%, Speed: 10960.67 record/sec
    FastEstimator: Converting Train TFRecords 60.0%, Speed: 11302.76 record/sec
    FastEstimator: Converting Train TFRecords 65.0%, Speed: 11152.31 record/sec
    FastEstimator: Converting Train TFRecords 70.0%, Speed: 10863.13 record/sec
    FastEstimator: Converting Train TFRecords 75.0%, Speed: 10893.36 record/sec
    FastEstimator: Converting Train TFRecords 80.0%, Speed: 11142.46 record/sec
    FastEstimator: Converting Train TFRecords 85.0%, Speed: 11371.03 record/sec
    FastEstimator: Converting Train TFRecords 90.0%, Speed: 11583.05 record/sec
    FastEstimator: Converting Train TFRecords 95.0%, Speed: 11784.00 record/sec
    FastEstimator: Reading non-empty directory: /root/fastestimator_data/adda/tfr
    FastEstimator: Found 60000 examples for train in /root/fastestimator_data/adda/tfr/train_summary1.json
    FastEstimator: Found 7291 examples for train in /root/fastestimator_data/adda/tfr/train_summary0.json


We can visualize an example output from the pipeline.


```python
import matplotlib
from matplotlib import pyplot as plt

plt.subplot(121)
plt.imshow(np.squeeze(a[0]["source_img"][1]), cmap='gray');
plt.axis('off');
plt.title('Sample MNIST Image');

plt.subplot(122)
plt.imshow(np.squeeze(a[0]["target_img"][3]), cmap='gray');
plt.axis('off');
plt.title('Sample USPS Image');
```


![png](assets/example/domain_adaptation/ADDA_files/ADDA_11_0.png)


## Network Definition

With ``Pipeline`` defined, we define the network architecture.
As we dicussed previously, the classification model is composed of the feature extraction network and the classifier network.
The training scheme is very similar to that of GAN; the objective is to train a feature extractor network for the USPS dataset so that the discriminator cannot reliably distinguish MNIST examples and USPS examples.
The feature extractor network for the USPS dataset is initialized from the feature extractor network for the MNIST dataset.


```python
from tensorflow.keras import layers, Model, Sequential

model_path = os.path.join(os.getcwd(), "feature_extractor.h5")

def build_feature_extractor(input_shape=(32, 32, 1), feature_dim=512):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(feature_dim, activation='relu'))        
    return model

def build_classifer(feature_dim=512, num_classes=10):
    model = Sequential()
    model.add(layers.Dense(num_classes, activation='softmax', input_dim=feature_dim))
    return model

def build_discriminator(feature_dim=512):
    model = Sequential()
    model.add(layers.Dense(1024, activation='relu', input_dim=feature_dim))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Step2: Define Network
feature_extractor = fe.build(model_def=build_feature_extractor,
                             model_name="fe",
                             loss_name="fe_loss",
                             optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9))

discriminator = fe.build(model_def=build_discriminator,
                         model_name="disc",
                         loss_name="d_loss",
                         optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9))
```

We need to define a ``TensorOp`` to extract a feature from MNIST images.
This feature will be used as an input to the discriminator.


```python
from fastestimator.op import TensorOp
from fastestimator.op.tensorop import Loss, ModelOp

class ExtractSourceFeature(TensorOp):
    def __init__(self, model_path, inputs, outputs=None, mode=None):
        super().__init__(inputs, outputs, mode)
        self.source_feature_extractor = tf.keras.models.load_model(model_path, compile=False)
        self.source_feature_extractor.trainable = False

    def forward(self, data, state):        
        return self.source_feature_extractor(data)
```

We define loss functions for the feature extractor network and the discriminator network.


```python
class FELoss(Loss):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        
    def forward(self, data, state):
        target_score = data        
        return self.cross_entropy(tf.ones_like(target_score), target_score)

class DLoss(Loss):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        
    def forward(self, data, state):
        source_score, target_score = data
        source_loss = self.cross_entropy(tf.ones_like(source_score), source_score)
        target_loss = self.cross_entropy(tf.zeros_like(target_score), target_score)
        total_loss = source_loss + target_loss
        return 0.5 * total_loss
```

We define the forward pass of the networks within one iteration of the training.


```python
network = fe.Network(ops=[
    ModelOp(inputs="target_img", outputs="target_feature", model=feature_extractor),
    ModelOp(inputs="target_feature", outputs="target_score", model=discriminator),
    ExtractSourceFeature(model_path=model_path, inputs="source_img", outputs="source_feature"),
    ModelOp(inputs="source_feature", outputs="source_score", model=discriminator),
    DLoss(inputs=("source_score", "target_score"), outputs="d_loss"),
    FELoss(inputs="target_score", outputs="fe_loss")
])
```

We need to define two ``Trace``:
* ``LoadPretrainedFE`` to load the weights of the feature extractor trained on MNIST
* ``EvaluateTargetClassifier`` to evaluate the classifier on the USPS dataset.

There are three key thins to keep in mind:
* The classifier network is never updated with any target label information
* Only the feature extractor is fine tuned to confuse the discriminator network
* The classifier only classifies on the basis of the output of the feature extractor network


```python
from fastestimator.trace import Trace

class LoadPretrainedFE(Trace):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
    def on_begin(self, state):
        self.network.model[self.model_name].load_weights('feature_extractor.h5')
        print("FastEstimator-LoadPretrainedFE: loaded pretrained feature extractor")

class EvaluateTargetClassifier(Trace):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.target_model = tf.keras.Sequential()
        self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        
    def on_begin(self, state):
        self.target_model.add(self.network.model[self.model_name])
        self.target_model.add(
            tf.keras.models.load_model("classifier.h5")
        )
    def on_batch_end(self, state):
        if state["epoch"] == 0 or state["epoch"] == 99:
            target_img, target_label = state["batch"]["target_img"], state["batch"]["target_label"]
            logits = self.target_model(target_img)
            self.acc_metric(target_label, logits)
    
    def on_epoch_end(self, state):
        if state["epoch"] == 0 or state["epoch"] == 99:
            acc = self.acc_metric.result()
            print("FastEstimator-EvaluateTargetClassifier: %0.4f at epoch %d" % (acc, state["epoch"]))
            self.acc_metric.reset_states()
```

## Defining Estimator

With ``Pipeline``, ``Network``, and ``Trace`` defined, we now define ``Estimator`` to put everything together


```python
traces = [LoadPretrainedFE(model_name="fe"),
          EvaluateTargetClassifier(model_name="fe")]
estimator = fe.Estimator(
    pipeline= pipeline, 
    network=network,
    traces = traces,
    epochs=100
)
```

We call ``fit`` method to start the training.


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator: Reading non-empty directory: /root/fastestimator_data/adda/tfr
    FastEstimator: Found 60000 examples for train in /root/fastestimator_data/adda/tfr/train_summary1.json
    FastEstimator: Found 7291 examples for train in /root/fastestimator_data/adda/tfr/train_summary0.json
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-LoadPretrainedFE: loaded pretrained feature extractor
    WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
    FastEstimator-Start: step: 0; total_train_steps: 5600; fe_lr: 1e-04; disc_lr: 1e-04; 
    FastEstimator-Train: step: 0; fe_loss: 0.6089596; d_loss: 0.681249; 
    FastEstimator-EvaluateTargetClassifier: 0.7874 at epoch 0
    FastEstimator-Train: step: 100; fe_loss: 1.0726796; d_loss: 0.4771829; examples/sec: 5513.1; progress: 1.8%; 
    FastEstimator-Train: step: 200; fe_loss: 1.1022232; d_loss: 0.4764578; examples/sec: 6433.9; progress: 3.6%; 
    FastEstimator-Train: step: 300; fe_loss: 1.2459202; d_loss: 0.4798668; examples/sec: 6221.2; progress: 5.4%; 
    FastEstimator-Train: step: 400; fe_loss: 1.3324102; d_loss: 0.4748652; examples/sec: 6457.0; progress: 7.1%; 
    FastEstimator-Train: step: 500; fe_loss: 1.5858265; d_loss: 0.5034922; examples/sec: 3345.9; progress: 8.9%; 
    FastEstimator-Train: step: 600; fe_loss: 0.7085456; d_loss: 0.5749392; examples/sec: 6861.5; progress: 10.7%; 
    FastEstimator-Train: step: 700; fe_loss: 1.7074094; d_loss: 0.5349842; examples/sec: 6895.7; progress: 12.5%; 
    FastEstimator-Train: step: 800; fe_loss: 1.5799985; d_loss: 0.482881; examples/sec: 6817.5; progress: 14.3%; 
    FastEstimator-Train: step: 900; fe_loss: 0.9060917; d_loss: 0.5182171; examples/sec: 8033.4; progress: 16.1%; 
    FastEstimator-Train: step: 1000; fe_loss: 1.0265226; d_loss: 0.4402258; examples/sec: 2913.7; progress: 17.9%; 
    FastEstimator-Train: step: 1100; fe_loss: 0.9295922; d_loss: 0.494996; examples/sec: 6929.2; progress: 19.6%; 
    FastEstimator-Train: step: 1200; fe_loss: 1.339618; d_loss: 0.5340438; examples/sec: 6848.2; progress: 21.4%; 
    FastEstimator-Train: step: 1300; fe_loss: 0.5638706; d_loss: 0.59447; examples/sec: 7950.7; progress: 23.2%; 
    FastEstimator-Train: step: 1400; fe_loss: 1.5171843; d_loss: 0.5671506; examples/sec: 6750.7; progress: 25.0%; 
    FastEstimator-Train: step: 1500; fe_loss: 1.52879; d_loss: 0.4776084; examples/sec: 3130.6; progress: 26.8%; 
    FastEstimator-Train: step: 1600; fe_loss: 1.1557212; d_loss: 0.5874342; examples/sec: 6832.0; progress: 28.6%; 
    FastEstimator-Train: step: 1700; fe_loss: 0.7100754; d_loss: 0.5629431; examples/sec: 7975.3; progress: 30.4%; 
    FastEstimator-Train: step: 1800; fe_loss: 1.2991563; d_loss: 0.5433497; examples/sec: 6720.4; progress: 32.1%; 
    FastEstimator-Train: step: 1900; fe_loss: 0.4490385; d_loss: 0.7224292; examples/sec: 2871.2; progress: 33.9%; 
    FastEstimator-Train: step: 2000; fe_loss: 1.1373507; d_loss: 0.5484131; examples/sec: 6895.8; progress: 35.7%; 
    FastEstimator-Train: step: 2100; fe_loss: 1.0352225; d_loss: 0.5430945; examples/sec: 7992.6; progress: 37.5%; 
    FastEstimator-Train: step: 2200; fe_loss: 1.522444; d_loss: 0.5251599; examples/sec: 6872.3; progress: 39.3%; 
    FastEstimator-Train: step: 2300; fe_loss: 1.4009509; d_loss: 0.508774; examples/sec: 6809.2; progress: 41.1%; 
    FastEstimator-Train: step: 2400; fe_loss: 1.5312254; d_loss: 0.5388322; examples/sec: 2896.9; progress: 42.9%; 
    FastEstimator-Train: step: 2500; fe_loss: 1.6519206; d_loss: 0.570011; examples/sec: 8011.6; progress: 44.6%; 
    FastEstimator-Train: step: 2600; fe_loss: 0.9349663; d_loss: 0.5017464; examples/sec: 6938.2; progress: 46.4%; 
    FastEstimator-Train: step: 2700; fe_loss: 1.1607829; d_loss: 0.4954276; examples/sec: 6897.4; progress: 48.2%; 
    FastEstimator-Train: step: 2800; fe_loss: 1.1784583; d_loss: 0.4916884; examples/sec: 6905.6; progress: 50.0%; 
    FastEstimator-Train: step: 2900; fe_loss: 1.2136245; d_loss: 0.5028952; examples/sec: 3363.6; progress: 51.8%; 
    FastEstimator-Train: step: 3000; fe_loss: 0.9984823; d_loss: 0.5247761; examples/sec: 6869.3; progress: 53.6%; 
    FastEstimator-Train: step: 3100; fe_loss: 0.9749998; d_loss: 0.496695; examples/sec: 6826.1; progress: 55.4%; 
    FastEstimator-Train: step: 3200; fe_loss: 0.9823316; d_loss: 0.4926038; examples/sec: 6785.6; progress: 57.1%; 
    FastEstimator-Train: step: 3300; fe_loss: 1.6716573; d_loss: 0.5352554; examples/sec: 3120.0; progress: 58.9%; 
    FastEstimator-Train: step: 3400; fe_loss: 1.0688219; d_loss: 0.5558775; examples/sec: 6889.5; progress: 60.7%; 
    FastEstimator-Train: step: 3500; fe_loss: 1.2807276; d_loss: 0.5147618; examples/sec: 6859.0; progress: 62.5%; 
    FastEstimator-Train: step: 3600; fe_loss: 1.3385832; d_loss: 0.5068592; examples/sec: 6872.5; progress: 64.3%; 
    FastEstimator-Train: step: 3700; fe_loss: 1.4010513; d_loss: 0.5224268; examples/sec: 8247.9; progress: 66.1%; 
    FastEstimator-Train: step: 3800; fe_loss: 1.0652959; d_loss: 0.5546493; examples/sec: 3066.8; progress: 67.9%; 
    FastEstimator-Train: step: 3900; fe_loss: 0.9134641; d_loss: 0.5320688; examples/sec: 6864.7; progress: 69.6%; 
    FastEstimator-Train: step: 4000; fe_loss: 0.7933622; d_loss: 0.5298726; examples/sec: 6904.5; progress: 71.4%; 
    FastEstimator-Train: step: 4100; fe_loss: 1.0647731; d_loss: 0.5019788; examples/sec: 6809.6; progress: 73.2%; 
    FastEstimator-Train: step: 4200; fe_loss: 1.4380388; d_loss: 0.5365941; examples/sec: 8224.2; progress: 75.0%; 
    FastEstimator-Train: step: 4300; fe_loss: 1.7495753; d_loss: 0.6115906; examples/sec: 3139.0; progress: 76.8%; 
    FastEstimator-Train: step: 4400; fe_loss: 1.1284056; d_loss: 0.4732235; examples/sec: 6933.8; progress: 78.6%; 
    FastEstimator-Train: step: 4500; fe_loss: 0.9639484; d_loss: 0.5253327; examples/sec: 6829.3; progress: 80.4%; 
    FastEstimator-Train: step: 4600; fe_loss: 0.9848607; d_loss: 0.464295; examples/sec: 8281.9; progress: 82.1%; 
    FastEstimator-Train: step: 4700; fe_loss: 0.8769747; d_loss: 0.4915569; examples/sec: 2876.2; progress: 83.9%; 
    FastEstimator-Train: step: 4800; fe_loss: 1.7685512; d_loss: 0.4833464; examples/sec: 6717.0; progress: 85.7%; 
    FastEstimator-Train: step: 4900; fe_loss: 1.7158316; d_loss: 0.5014951; examples/sec: 6947.6; progress: 87.5%; 
    FastEstimator-Train: step: 5000; fe_loss: 1.1106799; d_loss: 0.4509357; examples/sec: 8318.1; progress: 89.3%; 
    FastEstimator-Train: step: 5100; fe_loss: 1.0473665; d_loss: 0.4548874; examples/sec: 4789.3; progress: 91.1%; 
    FastEstimator-Train: step: 5200; fe_loss: 0.9096103; d_loss: 0.4814547; examples/sec: 1765.7; progress: 92.9%; 
    FastEstimator-Train: step: 5300; fe_loss: 1.0935009; d_loss: 0.5313207; examples/sec: 6374.3; progress: 94.6%; 
    FastEstimator-Train: step: 5400; fe_loss: 0.9431607; d_loss: 0.5197348; examples/sec: 7953.5; progress: 96.4%; 
    FastEstimator-Train: step: 5500; fe_loss: 0.4644586; d_loss: 0.6450016; examples/sec: 6866.9; progress: 98.2%; 
    FastEstimator-EvaluateTargetClassifier: 0.9402 at epoch 99
    FastEstimator-Finish: step: 5600; total_time: 134.23 sec; fe_lr: 1e-04; disc_lr: 1e-04; 


``EvaluateTargetClassifier`` outputs the classification accuracy on the USPS at the beginning of the training and the end of the training. We can observe significant improvement in the performance.


```python

```
