# Anomaly Detection with Fastestimator

In this notebook we will demonstrate how to do anomaly detection using one class classifier as described in [Adversarially Learned One-Class Classifier for Novelty Detection](https://arxiv.org/pdf/1802.09088.pdf). In real world, outliers or novelty class is often absent from the training dataset. Such problems can be efficiently modeled using one class classifiers.
In the algorihm demonstrated below, two networks are trained to compete with each other where one network acts as a novelty detector and other enhaces the inliers and distorts the outliers. We use images of digit "1" from MNIST dataset for training and images of other digits as outliers.


```python
import tempfile

import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.backend import binary_crossentropy
from fastestimator.op.numpyop import LambdaOp
from fastestimator.op.numpyop.univariate import ExpandDims, Normalize
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace import Trace
from fastestimator.trace.io import BestModelSaver
from fastestimator.util import to_number
from sklearn.metrics import auc, f1_score, roc_curve
from tensorflow.python.keras import layers
```


```python
# Parameters
epochs=20
batch_size=128
max_train_steps_per_epoch=None
save_dir=tempfile.mkdtemp()
```

## Building Components

### Downloading the data

First, we will download training images using tensorflow API. We will use images of digit `1` for training and test images of `1` as inliers and images of other digits as outliers. Outliers comprise 50% of our validation dataset.


```python
(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()

# Create Training Dataset
x_train, y_train = x_train[np.where((y_train == 1))], np.zeros(y_train[np.where((y_train == 1))].shape)
train_data = fe.dataset.NumpyDataset({"x": x_train, "y": y_train})

# Create Validation Dataset
x_eval0, y_eval0 = x_eval[np.where((y_eval == 1))], np.ones(y_eval[np.where((y_eval == 1))].shape)
x_eval1, y_eval1 = x_eval[np.where((y_eval != 1))], y_eval[np.where((y_eval != 1))]

# Ensuring outliers comprise 50% of the dataset
index = np.random.choice(x_eval1.shape[0], int(x_eval0.shape[0]), replace=False)
x_eval1, y_eval1 = x_eval1[index], np.zeros(y_eval1[index].shape)

x_eval, y_eval = np.concatenate([x_eval0, x_eval1]), np.concatenate([y_eval0, y_eval1])
eval_data = fe.dataset.NumpyDataset({"x": x_eval, "y": y_eval})
```

### Step 1: Create `Pipeline`

We will use the `LambdaOp` to add noise to the images during training.


```python
pipeline = fe.Pipeline(
    train_data=train_data,
    eval_data=eval_data,
    batch_size=batch_size,
    ops=[
        ExpandDims(inputs="x", outputs="x"),
        Normalize(inputs="x", outputs="x", mean=1.0, std=1.0, max_pixel_value=127.5),
        LambdaOp(fn=lambda x: x + np.random.normal(loc=0.0, scale=0.155, size=(28, 28, 1)),
                 inputs="x",
                 outputs="x_w_noise",
                 mode="train")
    ])
```

We can visualize sample images from our `Pipeline` using the 'get_results' method.


```python
sample_batch = pipeline.get_results()

img = fe.util.ImgData(Image=sample_batch["x"][0].numpy().reshape(1, 28, 28, 1), 
                      Noisy_Image=sample_batch["x_w_noise"][0].numpy().reshape(1, 28, 28, 1))
fig = img.paint_figure()
```


    
![png](assets/branches/r1.2/example/anomaly_detection/alocc_files/alocc_8_0.png)
    


### Step 2: Create `Network`

The architecture of our model consists of an Autoencoder (ecoder-decoder) network and a Discriminator network.
![Network Architecture](assets/branches/r1.2/example/anomaly_detection/network_architecture.PNG)[Credit: https://arxiv.org/pdf/1802.09088.pdf]


```python
def reconstructor(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential()
    # Encoder Block
    model.add(
        layers.Conv2D(32, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                      input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(
        layers.Conv2D(64, (5, 5),
                      strides=(2, 2),
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                      padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(
        layers.Conv2D(128, (5, 5),
                      strides=(2, 2),
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                      padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    # Decoder Block
    model.add(
        layers.Conv2DTranspose(32, (5, 5),
                               strides=(2, 2),
                               output_padding=(0, 0),
                               padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(
        layers.Conv2DTranspose(16, (5, 5),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(
        layers.Conv2DTranspose(1, (5, 5),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                               activation='tanh'))
    return model


def discriminator(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(16, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                      input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(
        layers.Conv2D(32, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(
        layers.Conv2D(64, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(
        layers.Conv2D(128, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model
```


```python
recon_model = fe.build(model_fn=reconstructor, optimizer_fn=lambda: tf.optimizers.RMSprop(2e-4), model_name="reconstructor")
disc_model = fe.build(model_fn=discriminator,
                      optimizer_fn=lambda: tf.optimizers.RMSprop(1e-4),
                      model_name="discriminator")
```

### Defining Loss

The losses of both the networks are smilar to a standarad GAN network with the exception of the autoencoder having and additional *reconstruction* loss term to enforce similarity between the input and the reconstructed image.
We first define custom `TensorOp`s to calculate the losses of both the networks.


```python
class RLoss(TensorOp):
    def __init__(self, alpha=0.2, inputs=None, outputs=None, mode=None):
        super().__init__(inputs, outputs, mode)
        self.alpha = alpha

    def forward(self, data, state):
        fake_score, x_fake, x = data
        recon_loss = binary_crossentropy(y_true=x, y_pred=x_fake, from_logits=True)
        adv_loss = binary_crossentropy(y_pred=fake_score, y_true=tf.ones_like(fake_score), from_logits=True)
        return adv_loss + self.alpha * recon_loss


class DLoss(TensorOp):
    def forward(self, data, state):
        true_score, fake_score = data
        real_loss = binary_crossentropy(y_pred=true_score, y_true=tf.ones_like(true_score), from_logits=True)
        fake_loss = binary_crossentropy(y_pred=fake_score, y_true=tf.zeros_like(fake_score), from_logits=True)
        total_loss = real_loss + fake_loss
        return total_loss
```

We now define the `Network` object:


```python
network = fe.Network(ops=[
    ModelOp(model=recon_model, inputs="x_w_noise", outputs="x_fake", mode="train"),
    ModelOp(model=recon_model, inputs="x", outputs="x_fake", mode="eval"),
    ModelOp(model=disc_model, inputs="x_fake", outputs="fake_score"),
    ModelOp(model=disc_model, inputs="x", outputs="true_score"),
    RLoss(inputs=("fake_score", "x_fake", "x"), outputs="rloss"),
    UpdateOp(model=recon_model, loss_name="rloss"),
    DLoss(inputs=("true_score", "fake_score"), outputs="dloss"),
    UpdateOp(model=disc_model, loss_name="dloss")
])
```

In this example we will also use the following traces:

1. BestModelSaver for saving the best model. For illustration purpose, we will save these models in a temporary directory.
2. A custom trace to calculate Area Under the Curve and F1-Score.


```python
class F1AUCScores(Trace):
    """Computes F1-Score and AUC Score for a classification task and reports it back to the logger.
    """
    def __init__(self, true_key, pred_key, mode=("eval", "test"), output_name=["auc_score", "f1_score"]):
        super().__init__(inputs=(true_key, pred_key), outputs=output_name, mode=mode)
        self.y_true = []
        self.y_pred = []

    @property
    def true_key(self):
        return self.inputs[0]

    @property
    def pred_key(self):
        return self.inputs[1]

    def on_epoch_begin(self, data):
        self.y_true = []
        self.y_pred = []

    def on_batch_end(self, data):
        y_true, y_pred = to_number(data[self.true_key]), to_number(data[self.pred_key])
        assert y_pred.size == y_true.size
        self.y_pred.extend(y_pred.ravel())
        self.y_true.extend(y_true.ravel())

    def on_epoch_end(self, data):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)
        eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr - fpr)))]
        y_pred_class = np.copy(self.y_pred)
        y_pred_class[y_pred_class >= eer_threshold] = 1
        y_pred_class[y_pred_class < eer_threshold] = 0
        f_score = f1_score(self.y_true, y_pred_class)

        data.write_with_log(self.outputs[0], roc_auc)
        data.write_with_log(self.outputs[1], f_score)
        

traces = [
    F1AUCScores(true_key="y", pred_key="fake_score", mode="eval", output_name=["auc_score", "f1_score"]),
    BestModelSaver(model=recon_model, save_dir=save_dir, metric='f1_score', save_best_mode='max', load_best_final=True),
    BestModelSaver(model=disc_model, save_dir=save_dir, metric='f1_score', save_best_mode='max', load_best_final=True)
]
```

### Step 3: Create `Estimator`


```python
estimator = fe.Estimator(pipeline=pipeline,
                         network=network,
                         epochs=epochs,
                         traces=traces,
                         max_train_steps_per_epoch=max_train_steps_per_epoch)
```

## Training


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 100; 
    FastEstimator-Train: step: 1; dloss: 1.4547124; rloss: 0.6044176; 
    FastEstimator-Train: step: 53; epoch: 1; epoch_time: 6.41 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/reconstructor_best_f1_score.h5
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/discriminator_best_f1_score.h5
    FastEstimator-Eval: step: 53; epoch: 1; dloss: 1.4323395; rloss: 0.6304608; auc_score: 0.758243707426886; f1_score: 0.6554770318021201; since_best_f1_score: 0; max_f1_score: 0.6554770318021201; 
    FastEstimator-Train: step: 100; dloss: 1.0820444; rloss: 0.72240007; steps/sec: 17.02; 
    FastEstimator-Train: step: 106; epoch: 2; epoch_time: 2.1 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/reconstructor_best_f1_score.h5
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/discriminator_best_f1_score.h5
    FastEstimator-Eval: step: 106; epoch: 2; dloss: 1.418492; rloss: 0.66902; auc_score: 0.8146631993634652; f1_score: 0.7268722466960352; since_best_f1_score: 0; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 159; epoch: 3; epoch_time: 2.1 sec; 
    FastEstimator-Eval: step: 159; epoch: 3; dloss: 1.4147544; rloss: 0.7020666; auc_score: 0.15153680451784432; f1_score: 0.2431718061674009; since_best_f1_score: 1; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 200; dloss: 1.0067211; rloss: 0.660446; steps/sec: 25.29; 
    FastEstimator-Train: step: 212; epoch: 4; epoch_time: 2.1 sec; 
    FastEstimator-Eval: step: 212; epoch: 4; dloss: 1.3572774; rloss: 0.7195197; auc_score: 0.019186089386559024; f1_score: 0.06696035242290749; since_best_f1_score: 2; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 265; epoch: 5; epoch_time: 2.1 sec; 
    FastEstimator-Eval: step: 265; epoch: 5; dloss: 1.1044953; rloss: 0.6936346; auc_score: 0.013531409497564477; f1_score: 0.0599647266313933; since_best_f1_score: 3; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 300; dloss: 1.0064102; rloss: 0.59952056; steps/sec: 25.26; 
    FastEstimator-Train: step: 318; epoch: 6; epoch_time: 2.1 sec; 
    FastEstimator-Eval: step: 318; epoch: 6; dloss: 1.0178115; rloss: 0.64911795; auc_score: 0.4008981350307594; f1_score: 0.4140969162995595; since_best_f1_score: 4; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 371; epoch: 7; epoch_time: 2.09 sec; 
    FastEstimator-Eval: step: 371; epoch: 7; dloss: 1.0237744; rloss: 0.6199698; auc_score: 0.37563857245434606; f1_score: 0.3857331571994716; since_best_f1_score: 5; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 400; dloss: 1.0064098; rloss: 0.5864141; steps/sec: 25.25; 
    FastEstimator-Train: step: 424; epoch: 8; epoch_time: 2.11 sec; 
    FastEstimator-Eval: step: 424; epoch: 8; dloss: 1.010116; rloss: 0.6061601; auc_score: 0.7393793786023405; f1_score: 0.6822183098591549; since_best_f1_score: 6; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 477; epoch: 9; epoch_time: 2.1 sec; 
    FastEstimator-Eval: step: 477; epoch: 9; dloss: 1.0113664; rloss: 0.6000464; auc_score: 0.790935589667954; f1_score: 0.7227112676056338; since_best_f1_score: 7; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 500; dloss: 1.0064089; rloss: 0.58218443; steps/sec: 25.19; 
    FastEstimator-Train: step: 530; epoch: 10; epoch_time: 2.11 sec; 
    FastEstimator-Eval: step: 530; epoch: 10; dloss: 1.0653309; rloss: 0.59754586; auc_score: 0.37388422053600884; f1_score: 0.39136183340678715; since_best_f1_score: 8; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 583; epoch: 11; epoch_time: 2.11 sec; 
    FastEstimator-Eval: step: 583; epoch: 11; dloss: 1.0164112; rloss: 0.5950321; auc_score: 0.5598245648081662; f1_score: 0.5395842547545334; since_best_f1_score: 9; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 600; dloss: 1.006409; rloss: 0.5809828; steps/sec: 25.18; 
    FastEstimator-Train: step: 636; epoch: 12; epoch_time: 2.11 sec; 
    FastEstimator-Eval: step: 636; epoch: 12; dloss: 1.0633637; rloss: 0.5527591; auc_score: 0.7476892623571193; f1_score: 0.678996036988111; since_best_f1_score: 10; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 689; epoch: 13; epoch_time: 2.1 sec; 
    FastEstimator-Eval: step: 689; epoch: 13; dloss: 1.2391297; rloss: 0.6055295; auc_score: 0.2452661608026548; f1_score: 0.2874779541446208; since_best_f1_score: 11; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 700; dloss: 1.0498809; rloss: 0.57184714; steps/sec: 25.21; 
    FastEstimator-Train: step: 742; epoch: 14; epoch_time: 2.1 sec; 
    FastEstimator-Eval: step: 742; epoch: 14; dloss: 1.1737943; rloss: 0.60413545; auc_score: 0.4565204059849794; f1_score: 0.4269960299955889; since_best_f1_score: 12; max_f1_score: 0.7268722466960352; 
    FastEstimator-Train: step: 795; epoch: 15; epoch_time: 2.11 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/reconstructor_best_f1_score.h5
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/discriminator_best_f1_score.h5
    FastEstimator-Eval: step: 795; epoch: 15; dloss: 1.0669132; rloss: 0.60187024; auc_score: 0.9049614780026781; f1_score: 0.8089788732394366; since_best_f1_score: 0; max_f1_score: 0.8089788732394366; 
    FastEstimator-Train: step: 800; dloss: 1.08307; rloss: 0.5877172; steps/sec: 25.13; 
    FastEstimator-Train: step: 848; epoch: 16; epoch_time: 2.11 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/reconstructor_best_f1_score.h5
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/discriminator_best_f1_score.h5
    FastEstimator-Eval: step: 848; epoch: 16; dloss: 1.3530512; rloss: 0.6081449; auc_score: 0.941642958334142; f1_score: 0.8656979304271246; since_best_f1_score: 0; max_f1_score: 0.8656979304271246; 
    FastEstimator-Train: step: 900; dloss: 1.3662006; rloss: 0.58985895; steps/sec: 25.06; 
    FastEstimator-Train: step: 901; epoch: 17; epoch_time: 2.14 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/reconstructor_best_f1_score.h5
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/discriminator_best_f1_score.h5
    FastEstimator-Eval: step: 901; epoch: 17; dloss: 1.227813; rloss: 0.5782926; auc_score: 0.9882008189563158; f1_score: 0.948526176858777; since_best_f1_score: 0; max_f1_score: 0.948526176858777; 
    FastEstimator-Train: step: 954; epoch: 18; epoch_time: 2.11 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/reconstructor_best_f1_score.h5
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpf8gmdf9j/discriminator_best_f1_score.h5
    FastEstimator-Eval: step: 954; epoch: 18; dloss: 1.328288; rloss: 0.5912411; auc_score: 0.9875805856896116; f1_score: 0.9559471365638766; since_best_f1_score: 0; max_f1_score: 0.9559471365638766; 
    FastEstimator-Train: step: 1000; dloss: 1.2895771; rloss: 0.53317624; steps/sec: 24.93; 
    FastEstimator-Train: step: 1007; epoch: 19; epoch_time: 2.11 sec; 
    FastEstimator-Eval: step: 1007; epoch: 19; dloss: 1.3832934; rloss: 0.6033389; auc_score: 0.9015963826194958; f1_score: 0.813215859030837; since_best_f1_score: 1; max_f1_score: 0.9559471365638766; 
    FastEstimator-Train: step: 1060; epoch: 20; epoch_time: 2.11 sec; 
    FastEstimator-Eval: step: 1060; epoch: 20; dloss: 1.3230872; rloss: 0.5129401; auc_score: 0.4634978361699238; f1_score: 0.48722466960352423; since_best_f1_score: 2; max_f1_score: 0.9559471365638766; 
    FastEstimator-BestModelSaver: Restoring model from /tmp/tmpf8gmdf9j/reconstructor_best_f1_score.h5
    FastEstimator-BestModelSaver: Restoring model from /tmp/tmpf8gmdf9j/discriminator_best_f1_score.h5
    FastEstimator-Finish: step: 1060; total_time: 62.06 sec; discriminator_lr: 1e-04; reconstructor_lr: 0.0002; 


## Inferencing

Once the training is finished, we will apply the model to visualize the reconstructed image of the inliers and outliers.


```python
idx0 = np.random.randint(len(x_eval0))
idx1 = np.random.randint(len(x_eval1))

data = [{"x": x_eval0[idx0]}, {"x": x_eval1[idx1]}]
result = [pipeline.transform(data[i], mode="infer") for i in range(len(data))]
```


```python
network = fe.Network(ops=[
    ModelOp(model=recon_model, inputs="x", outputs="x_fake"),
    ModelOp(model=disc_model, inputs="x_fake", outputs="fake_score")
])

output_imgs = [network.transform(result[i], mode="infer") for i in range(len(result))]
```


```python
base_image = output_imgs[0]["x"].numpy()
anomaly_image = output_imgs[1]["x"].numpy()

recon_base_image = output_imgs[0]["x_fake"].numpy()
recon_anomaly_image = output_imgs[1]["x_fake"].numpy()

img1 = fe.util.ImgData(Input_Image=base_image, Reconstructed_Image=recon_base_image)
fig1 = img1.paint_figure()

img2 = fe.util.ImgData(Input_Image=anomaly_image, Reconstructed_Image=recon_anomaly_image)
fig2 = img2.paint_figure()
```


    
![png](assets/branches/r1.2/example/anomaly_detection/alocc_files/alocc_25_0.png)
    



    
![png](assets/branches/r1.2/example/anomaly_detection/alocc_files/alocc_25_1.png)
    


Note that the network is trained on inliers, so it's able to properly reconstruct them but does a poor job at reconstructing the outliers, thereby making it easier for discriminator to detect the outliers. 
