# Horse to Zebra Unpaired Image Translation with CycleGAN in FastEstimator

This notebook demonstrates how to perform an unpaired image to image translation using CycleGAN in FastEstimator.
The details of the method is found in [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).
We will specifically look at the problem of translating horse images to zebra images.


```python
import os
import cv2
import time

import tensorflow as tf
import fastestimator as fe

import matplotlib
from matplotlib import pyplot as plt
```

## Step 1: Defining Input Pipeline

First, we will download the dataset of horses and zebras via our dataset API.
The images will be first downloaded from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).
Then, two csv files containing relative paths to these images as *trainA.csv* and *trainB.csv*.
The root path of the downloaded images will be *parent_path*.


```python
from fastestimator.dataset.horse2zebra import load_data
trainA_csv, trainB_csv, _, _, parent_path = load_data()
```

Once the images are downloaded, we will create *tfrecords* that will be used extensively by *Pipeline*.
*RecordWriter* will create tfrecords using the csv files as an input; *ImageReader* operator will take each row in the csv file to read images. The tfrecords will be saved under a folder named *tfrecords*.


```python
from fastestimator.op.numpyop import ImageReader
from fastestimator.util import RecordWriter

tfr_save_dir = os.path.join(parent_path, 'tfrecords')
writer = RecordWriter(
    train_data=(trainA_csv, trainB_csv),                                                                          
    save_dir=tfr_save_dir,                                                                                        
    ops=([ImageReader(inputs="imgA", outputs="imgA", parent_path=parent_path)],                                   
         [ImageReader(inputs="imgB", outputs="imgB", parent_path=parent_path)]))       
```

We need to define two custom operators to preprocess input images:
1. Rescaling the pixel values to be between -1 and 1
2. Random image augmentation applying a random jitter followed by a random horizontal flip as described in the [paper](https://arxiv.org/abs/1703.10593).
Each operator is defined by inheriting from base class ``fastestimator.util.op.TensorOp``.
The data transform function is defined within ``forward`` method.


```python
from fastestimator.op import TensorOp

class Myrescale(TensorOp):
    def forward(self, data, state):
        data = tf.cast(data, tf.float32)
        data = (data - 127.5) / 127.5
        return data
    
class RandomAug(TensorOp):
    def forward(self, data, state):
        # resizing to 286 x 286 x 3
        data = tf.image.resize(data, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        data = tf.image.random_crop(data, size=[256, 256, 3]) 

        # random mirroring
        data = tf.image.random_flip_left_right(data)

        return data
```


```python
#Parameters
epochs = 50
batch_size = 1
steps_per_epoch = None
validation_steps = None
```

Given these operators and *tfrecord* writer, we can now define `Pipeline` object.


```python
pipeline = fe.Pipeline(
    data=writer,
    batch_size=batch_size,
    ops=[
        Myrescale(inputs="imgA", outputs="imgA"),
        RandomAug(inputs="imgA", outputs="real_A"),
        Myrescale(inputs="imgB", outputs="imgB"),
        RandomAug(inputs="imgB", outputs="real_B")
    ])
```

We can visualize sample images from the ``pipeline`` using ``show_results`` method.
``show_results`` returns a list of batch data.
By default, it only returns only one batch of data.

For visualization purpose only, we normalize the pixel values to between 0 and 1.
Because ``ImageReader`` operator internally uses ``cv2.imread``, we need to perform color transformation for proper visualization.


```python
def normalize_and_convert_color(img):
    img = img.numpy()
    img = (img + 1) * 0.5
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
```


```python
sample_batch = pipeline.show_results()
horse_img = sample_batch[0]["real_A"]
horse_img = normalize_and_convert_color(horse_img[0])

zebra_img = sample_batch[0]["real_B"]
zebra_img = normalize_and_convert_color(zebra_img[0])

plt.subplot(121)
plt.title('Sample Horse Image')
plt.imshow(horse_img);
plt.axis('off');

plt.subplot(122)
plt.title('Sample Zebra Image')
plt.imshow(zebra_img);
plt.axis('off');

plt.tight_layout()
```

    FastEstimator: Reading non-empty directory: /tmp/.fe/HORSE2ZEBRA/FEdata
    FastEstimator: Found 1334 examples for train in /tmp/.fe/HORSE2ZEBRA/FEdata/train_summary1.json
    FastEstimator: Found 1067 examples for train in /tmp/.fe/HORSE2ZEBRA/FEdata/train_summary0.json



![png](cyclegan_files/cyclegan_14_1.png)


## Step 2: Defining Model Architectures

In CycleGAN, there are 2 generators and 2 discriminators being trained.
* Generator `g_AtoB` learns to map horse images to zebra images
* Generator `g_BtoA` learns to map zebra images to horse images
* Discriminator `d_A` learns to differentiate between real hores images and fake horse images produced by `g_BtoA`
* Discriminator `d_B` learns to differentiate between image zebra and fake zebra images produced by `g_AtoB`

We first create `FEModel` instances for each model by specifying the following:
* model definition
* model name
* loss name
* optimizer

The architecture of generator is a modified resnet, and the architecture of discriminator is a PatchGAN.


```python
from fastestimator.architecture.cyclegan import build_generator, build_discriminator

g_AtoB = fe.build(model_def=build_generator,
                 model_name="g_AtoB",
                 loss_name="g_AtoB_loss",
                 optimizer=tf.keras.optimizers.Adam(2e-4, 0.5))

g_BtoA = fe.build(model_def=build_generator,
                 model_name="g_BtoA",
                 loss_name="g_BtoA_loss",
                 optimizer=tf.keras.optimizers.Adam(2e-4, 0.5))

d_A = fe.build(model_def=build_discriminator,
              model_name="d_A",
              loss_name="d_A_loss",
              optimizer=tf.keras.optimizers.Adam(2e-4, 0.5))

d_B = fe.build(model_def=build_discriminator,
              model_name="d_B",
              loss_name="d_B_loss",
              optimizer=tf.keras.optimizers.Adam(2e-4, 0.5))
```

## Loss functions
With `FEModel` being defined for each network, we need to define associated losses.
We provide `Loss` class from which users can derive to express necessary logics for calculating the loss.
In FastEstimator, the computation of loss is yet another `TensorOp`; therefore, the final loss value needs to be returned in `forward` method.

Because horse images and zebra images are unpaired, the loss of generator is quite complex. 
The generator's loss is composed of tree terms: 1) adversarial; 2) cycle-consistency; 3) identity.
The cycle-consistency term and identity term are weighted by a parameter `LAMBDA`. In the paper the authors used 10 for `LAMBDA`.

Let's consider computing the loss for `g_AtoB` which translates horses to zebras.
1. Adversarial term that is computed as binary cross entropy between 1s and `d_A`'s prediction on the translated images
2. Cycle consistency term is computed with mean absolute error between original *horse* images and the cycled horse images that are translated *forward* by `g_AtoB` and then *backward* by `g_BtoA`.
3. Identity term that is computed with the mean absolute error between original *zebra* and the output of `g_AtoB` on these images.

The discriminator's loss is the standard adversarial loss that is computed as binary cross entropy between:
* 1s and real images
* 0s and fake images


```python
from fastestimator.op.tensorop import Loss
LAMBDA = 10
class GLoss(Loss):
    def __init__(self, inputs, weight, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)
        self.LAMBDA = weight

    def _adversarial_loss(self, fake_img):
        return tf.reduce_mean(self.cross_entropy(tf.ones_like(fake_img), fake_img), axis=(1, 2)) 

    def _identity_loss(self, real_img, same_img):
        return 0.5 * self.LAMBDA * tf.reduce_mean(tf.abs(real_img - same_img), axis=(1, 2, 3)) 

    def _cycle_loss(self, real_img, cycled_img):
        return self.LAMBDA * tf.reduce_mean(tf.abs(real_img - cycled_img), axis=(1, 2, 3)) 

    def forward(self, data, state):
        real_img, fake_img, cycled_img, same_img = data
        total_loss = self._adversarial_loss(fake_img) + self._identity_loss(real_img, same_img) + self._cycle_loss(
            real_img, cycled_img)
        return total_loss

class DLoss(Loss):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)

    def forward(self, data, state):
        real_img, fake_img = data
        real_img_loss = tf.reduce_mean(self.cross_entropy(tf.ones_like(real_img), real_img), axis=(1, 2))
        fake_img_loss = tf.reduce_mean(self.cross_entropy(tf.zeros_like(real_img), fake_img), axis=(1, 2))
        total_loss = real_img_loss + fake_img_loss
        return 0.5 * total_loss

```

Once associated losses are defined, we now have to define `Network` object which contains a sequence of operators `ModelOp`. This chain of `ModelOp`s followed by `Loss` defines a forward pass of batch of data throughout networks.


```python
from fastestimator.op.tensorop import ModelOp

network = fe.Network(ops=[
    ModelOp(inputs="real_A", model=g_AtoB, outputs="fake_B"),
    ModelOp(inputs="real_B", model=g_BtoA, outputs="fake_A"),
    ModelOp(inputs="real_A", model=d_A, outputs="d_real_A"),
    ModelOp(inputs="fake_A", model=d_A, outputs="d_fake_A"),
    ModelOp(inputs="real_B", model=d_B, outputs="d_real_B"),
    ModelOp(inputs="fake_B", model=d_B, outputs="d_fake_B"),
    ModelOp(inputs="real_A", model=g_BtoA, outputs="same_A"),
    ModelOp(inputs="fake_B", model=g_BtoA, outputs="cycled_A"),
    ModelOp(inputs="real_B", model=g_AtoB, outputs="same_B"),
    ModelOp(inputs="fake_A", model=g_AtoB, outputs="cycled_B"),
    GLoss(inputs=("real_A", "d_fake_B", "cycled_A", "same_A"), weight=LAMBDA, outputs="g_AtoB_loss"),
    GLoss(inputs=("real_B", "d_fake_A", "cycled_B", "same_B"), weight=LAMBDA, outputs="g_BtoA_loss"),
    DLoss(inputs=("d_real_A", "d_fake_A"), outputs="d_A_loss"),
    DLoss(inputs=("d_real_B", "d_fake_B"), outputs="d_B_loss")
])
```

In this example we will also use `ModelSaver` traces to save the two generators `g_AtoB` and `g_BtoA` throughout training. For illustration purpose, we will save these models in temporary directory


```python
from fastestimator.trace import ModelSaver
import tempfile
model_dir=tempfile.mkdtemp()
os.makedirs(model_dir, exist_ok=True)
traces = [
    ModelSaver(model_name="g_AtoB", save_dir=model_dir, save_freq=10),
    ModelSaver(model_name="g_BtoA", save_dir=model_dir, save_freq=10)
]
```

Finally, we are ready to define `Estimator` object and then call `fit` method to start the training.
Just for the sake of demo purpose, we would only run 50 epochs.


```python
estimator = fe.Estimator(network=network, 
                         pipeline=pipeline, 
                         epochs=epochs, 
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps,
                         traces=traces)
estimator.fit()
```

Below are infering results of the two generators. 


```python
horse_img_t = tf.convert_to_tensor(horse_img)
horse_img_t = tf.expand_dims(horse_img_t, axis=0)
zebra_img_t = tf.convert_to_tensor(zebra_img)
zebra_img_t = tf.expand_dims(zebra_img_t, axis=0)
fake_zebra = estimator.network.model["g_AtoB"](horse_img_t)
fake_horse = estimator.network.model["g_BtoA"](zebra_img_t)
```


```python
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.imshow(horse_img);
plt.title('Real Horse Image')
plt.axis('off');

plt.subplot(222)
fake_zebra_display = normalize_and_convert_color(fake_zebra[0])
plt.imshow(fake_zebra_display);
plt.title('Translated Zebra Image')
plt.axis('off');

plt.subplot(223)
plt.imshow(zebra_img)
plt.title('Real Zebra Image')
plt.axis('off');

plt.subplot(224)
fake_horse_display = normalize_and_convert_color(fake_horse[0])
plt.imshow(fake_horse_display);
plt.title('Translated Horse Image')
plt.axis('off');

plt.tight_layout()
```


![png](cyclegan_files/cyclegan_29_0.png)


Note the addition of zebra-like stripe texture on top of horses when translating from horses to zebras.
When translating zebras to horses, we can observe that generator removes the stripe texture from zebras.
