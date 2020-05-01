# Fast Style Transfer with FastEstimator

In this notebook we will demonstrate how to do a neural image style transfer with perceptual loss as described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf).
Typical neural style transfer involves two images, an image containing semantics that you want to preserve and another image serving as a reference style; the first image is often referred as *content image* and the other image as *style image*.
In [paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf) training images of COCO2014 dataset are used to learn the style transfer from any content image.


```python
import tempfile
import os
import cv2
import tensorflow as tf
import numpy as np

import fastestimator as fe
from fastestimator.backend import reduce_mean
from fastestimator.op.numpyop.multivariate import Resize
from fastestimator.op.numpyop.univariate import Normalize, ReadImage
from fastestimator.trace.io import ModelSaver

import matplotlib
from matplotlib import pyplot as plt
```


```python
#Parameters
batch_size = 4
epochs = 2
max_steps_per_epoch = None
log_steps = 2000
style_weight=5.0
content_weight=1.0
tv_weight=1e-4
save_dir = tempfile.mkdtemp()
style_img_path = 'Vassily_Kandinsky,_1913_-_Composition_7.jpg'
test_img_path = 'panda.jpeg'
data_dir = None
```

In this notebook we will use *Vassily Kandinsky's Composition 7* as a style image.
We will also resize the style image to $256 \times 256$ to make the dimension consistent with that of COCO images.


```python
style_img = cv2.imread(style_img_path)
style_img = cv2.resize(style_img, (256, 256))
style_img = (style_img.astype(np.float32) - 127.5) / 127.5
style_img_t = tf.convert_to_tensor(np.expand_dims(style_img, axis=0))

style_img_disp = cv2.cvtColor((style_img + 1) * 0.5, cv2.COLOR_BGR2RGB)
plt.imshow(style_img_disp)
plt.title('Vassily Kandinsky\'s Composition 7')
plt.axis('off');
```


![png](assets/example/style_transfer/fst_files/fst_4_0.png)


## Building Components

### Downloading the data

First, we will download training images of COCO2014 dataset via our dataset API. Downloading the images will take awhile.


```python
from fastestimator.dataset.data import mscoco
train_data, _ = mscoco.load_data(root_dir=data_dir, load_bboxes=False, load_masks=False, load_captions=False)
```

### Step 1: Create pipeline


```python
pipeline = fe.Pipeline(
    train_data=train_data,
    batch_size=batch_size,
    ops=[
        ReadImage(inputs="image", outputs="image"),
        Normalize(inputs="image", outputs="image", mean=1.0, std=1.0, max_pixel_value=127.5),
        Resize(height=256, width=256, image_in="image", image_out="image")
    ])
```

We can visualize sample images from pipeline using get_results method.


```python
import matplotlib.pyplot as plt
import numpy as np

def Minmax(data):
    data_max = np.max(data)
    data_min = np.min(data)
    data = (data - data_min) / max((data_max - data_min), 1e-7)
    return data

sample_batch = pipeline.get_results()
img = Minmax(sample_batch["image"][0].numpy())
plt.imshow(img)
plt.show()
```


![png](assets/example/style_transfer/fst_files/fst_10_0.png)


### Step 2: Create network

The architecture of the model is a modified resnet.


```python
from typing import Dict, List, Tuple, Union

import tensorflow as tf

from fastestimator.layers.tensorflow import InstanceNormalization, ReflectionPadding2D


def _residual_block(x0, num_filter, kernel_size=(3, 3), strides=(1, 1)):
    initializer = tf.random_normal_initializer(0., 0.02)
    x0_cropped = tf.keras.layers.Cropping2D(cropping=2)(x0)

    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               kernel_initializer=initializer)(x0)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               kernel_initializer=initializer)(x)

    x = InstanceNormalization()(x)
    x = tf.keras.layers.Add()([x, x0_cropped])
    return x


def _conv_block(x0, num_filter, kernel_size=(9, 9), strides=(1, 1), padding="same", apply_relu=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               kernel_initializer=initializer)(x0)

    x = InstanceNormalization()(x)
    if apply_relu:
        x = tf.keras.layers.ReLU()(x)
    return x


def _upsample(x0, num_filter, kernel_size=(3, 3), strides=(2, 2), padding="same"):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2DTranspose(filters=num_filter,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        kernel_initializer=initializer)(x0)

    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def _downsample(x0, num_filter, kernel_size=(3, 3), strides=(2, 2), padding="same"):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(filters=num_filter,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               kernel_initializer=initializer)(x0)

    x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def StyleTransferNet(input_shape=(256, 256, 3), num_resblock=5):
    """Creates the Style Transfer Network.
    """
    x0 = tf.keras.layers.Input(shape=input_shape)
    x = ReflectionPadding2D(padding=(40, 40))(x0)
    x = _conv_block(x, num_filter=32)
    x = _downsample(x, num_filter=64)
    x = _downsample(x, num_filter=128)

    for _ in range(num_resblock):
        x = _residual_block(x, num_filter=128)

    x = _upsample(x, num_filter=64)
    x = _upsample(x, num_filter=32)
    x = _conv_block(x, num_filter=3, apply_relu=False)
    x = tf.keras.layers.Activation("tanh")(x)
    return tf.keras.Model(inputs=x0, outputs=x)


def LossNet(input_shape=(256, 256, 3),
            style_layers=["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"],
            content_layers=["block3_conv3"]):
    """Creates the network to compute the style loss.
    This network outputs a dictionary with outputs values for style and content, based on a list of layers from VGG16
    for each.
    """
    x0 = tf.keras.layers.Input(shape=input_shape)
    mdl = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=x0)
    # Compute style loss
    style_output = [mdl.get_layer(name).output for name in style_layers]
    content_output = [mdl.get_layer(name).output for name in content_layers]
    output = {"style": style_output, "content": content_output}
    return tf.keras.Model(inputs=x0, outputs=output)
```


```python
model = fe.build(model_fn=StyleTransferNet, 
                 model_names="style_transfer_net",
                 optimizer_fn=lambda: tf.optimizers.Adam(1e-3))
```

### Defining Loss

The perceptual loss described in the [paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf) is computed based on intermediate layers of VGG16 pretrained on ImageNet; specifically, `relu1_2`, `relu2_2`, `relu3_3`, and `relu4_3` of VGG16 are used.
The *style* loss term is computed as the squared l2 norm of the difference in Gram Matrix of these feature maps between an input image and the reference stlye image.
The *content* loss is simply l2 norm of the difference in `relu3_3` of the input image and the reference style image.
In addition, the method also uses total variation loss to enforce spatial smoothness in the output image.
The final loss is weighted sum of the style loss term, the content loss term (feature reconstruction term in the [paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)), and the total variation term.

We first define a custom `TensorOp` that outputs intermediate layers of VGG16.
Given these intermediate layers returned by the loss network as a dictionary, we define a custom `StyleContentLoss` class that encapsulates all the logic of the loss calculation.


```python
from fastestimator.op.tensorop import TensorOp

class ExtractVGGFeatures(TensorOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs, outputs, mode)
        self.vgg = LossNet()

    def forward(self, data, state):
        return self.vgg(data)


class StyleContentLoss(TensorOp):
    def __init__(self, style_weight, content_weight, tv_weight, inputs, outputs=None, mode=None, average_loss=True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        self.average_loss = average_loss

    def calculate_style_recon_loss(self, y_true, y_pred):
        y_true_gram = self.calculate_gram_matrix(y_true)
        y_pred_gram = self.calculate_gram_matrix(y_pred)
        y_diff_gram = y_pred_gram - y_true_gram
        y_norm = tf.math.sqrt(tf.reduce_sum(tf.math.square(y_diff_gram), axis=(1, 2)))
        return y_norm

    def calculate_feature_recon_loss(self, y_true, y_pred):
        y_diff = y_pred - y_true
        num_elts = tf.cast(tf.reduce_prod(y_diff.shape[1:]), tf.float32)
        y_diff_norm = tf.reduce_sum(tf.square(y_diff), axis=(1, 2, 3)) / num_elts
        return y_diff_norm

    def calculate_gram_matrix(self, x):
        x = tf.cast(x, tf.float32)
        num_elts = tf.cast(x.shape[1] * x.shape[2] * x.shape[3], tf.float32)
        gram_matrix = tf.einsum('bijc,bijd->bcd', x, x)
        gram_matrix /= num_elts
        return gram_matrix

    def calculate_total_variation(self, y_pred):
        return tf.image.total_variation(y_pred)

    def forward(self, data, state):
        y_pred, y_style, y_content, image_out = data

        style_loss = [self.calculate_style_recon_loss(a, b) for a, b in zip(y_style['style'], y_pred['style'])]
        style_loss = tf.add_n(style_loss)
        style_loss *= self.style_weight

        content_loss = [
            self.calculate_feature_recon_loss(a, b) for a, b in zip(y_content['content'], y_pred['content'])
        ]
        content_loss = tf.add_n(content_loss)
        content_loss *= self.content_weight

        total_variation_reg = self.calculate_total_variation(image_out)
        total_variation_reg *= self.tv_weight
        loss = style_loss + content_loss + total_variation_reg

        if self.average_loss:
            loss = reduce_mean(loss)

        return loss
```

We now define the network object


```python
from fastestimator.op.tensorop.model import ModelOp, UpdateOp

network = fe.Network(ops=[
    ModelOp(inputs="image", model=model, outputs="image_out"),
    ExtractVGGFeatures(inputs=lambda: style_img_t, outputs="y_style"),
    ExtractVGGFeatures(inputs="image", outputs="y_content"),
    ExtractVGGFeatures(inputs="image_out", outputs="y_pred"),
    StyleContentLoss(style_weight=style_weight,
                     content_weight=content_weight,
                     tv_weight=tv_weight,
                     inputs=('y_pred', 'y_style', 'y_content', 'image_out'),
                     outputs='loss'),
    UpdateOp(model=model, loss_name="loss")
])
```

### Step 3: Estimator

We can now define `Estimator`. We will use `Trace` to save intermediate models.


```python
estimator = fe.Estimator(network=network,
                         pipeline=pipeline,
                         traces=ModelSaver(model=model, save_dir=save_dir, frequency=1),
                         epochs=epochs,
                         max_steps_per_epoch=max_steps_per_epoch,
                         log_steps=log_steps)
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
                                                                            
    
    FastEstimator-Start: step: 1; style_transfer_net_lr: 0.001; 
    FastEstimator-Train: step: 1; loss: 686.76025; 
    FastEstimator-Train: step: 2000; loss: 194.85736; steps/sec: 10.82; 
    FastEstimator-Train: step: 4000; loss: 172.5411; steps/sec: 10.81; 
    FastEstimator-Train: step: 6000; loss: 173.80026; steps/sec: 10.81; 
    FastEstimator-Train: step: 8000; loss: 168.31807; steps/sec: 10.81; 
    FastEstimator-Train: step: 10000; loss: 169.65088; steps/sec: 10.81; 
    FastEstimator-Train: step: 12000; loss: 157.52707; steps/sec: 10.81; 
    FastEstimator-Train: step: 14000; loss: 157.95462; steps/sec: 10.82; 
    FastEstimator-Train: step: 16000; loss: 156.0791; steps/sec: 10.82; 
    FastEstimator-Train: step: 18000; loss: 141.07487; steps/sec: 10.82; 
    FastEstimator-Train: step: 20000; loss: 165.1513; steps/sec: 10.82; 
    FastEstimator-Train: step: 22000; loss: 142.25858; steps/sec: 10.82; 
    FastEstimator-Train: step: 24000; loss: 141.33316; steps/sec: 10.82; 
    FastEstimator-Train: step: 26000; loss: 136.16362; steps/sec: 10.82; 
    FastEstimator-Train: step: 28000; loss: 159.05832; steps/sec: 10.82; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpyby4rzao/style_transfer_net_epoch_1.h5
    FastEstimator-Train: step: 29572; epoch: 1; epoch_time: 2746.03 sec; 
    FastEstimator-Train: step: 30000; loss: 140.72166; steps/sec: 10.52; 
    FastEstimator-Train: step: 32000; loss: 153.47067; steps/sec: 10.82; 
    FastEstimator-Train: step: 34000; loss: 157.17432; steps/sec: 10.82; 
    FastEstimator-Train: step: 36000; loss: 145.79706; steps/sec: 10.82; 
    FastEstimator-Train: step: 38000; loss: 152.90091; steps/sec: 10.81; 
    FastEstimator-Train: step: 40000; loss: 146.31168; steps/sec: 10.81; 
    FastEstimator-Train: step: 42000; loss: 148.24469; steps/sec: 10.82; 
    FastEstimator-Train: step: 44000; loss: 149.6113; steps/sec: 10.82; 
    FastEstimator-Train: step: 46000; loss: 136.75755; steps/sec: 10.82; 
    FastEstimator-Train: step: 48000; loss: 149.79001; steps/sec: 10.82; 
    FastEstimator-Train: step: 50000; loss: 144.61955; steps/sec: 10.82; 
    FastEstimator-Train: step: 52000; loss: 138.5368; steps/sec: 10.82; 
    FastEstimator-Train: step: 54000; loss: 144.22594; steps/sec: 10.82; 
    FastEstimator-Train: step: 56000; loss: 140.38748; steps/sec: 10.82; 
    FastEstimator-Train: step: 58000; loss: 150.7169; steps/sec: 10.82; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpyby4rzao/style_transfer_net_epoch_2.h5
    FastEstimator-Train: step: 59144; epoch: 2; epoch_time: 2734.43 sec; 
    FastEstimator-Finish: step: 59144; total_time: 5480.64 sec; style_transfer_net_lr: 0.001; 


## Inferencing

Once the training is finished, we will apply the model to perform the style transfer on arbitrary images. Here we use a photo of a panda.


```python
data = {"image":test_img_path}
result = pipeline.transform(data, mode="infer")
test_img = np.squeeze(result["image"])
```


```python
network = fe.Network(ops=[
    ModelOp(inputs='image', model=model, outputs="image_out")
])

predictions = network.transform(result, mode="infer")
output_img = np.squeeze(predictions["image_out"])
```


```python
output_img_disp = (output_img + 1) * 0.5
test_img_disp = (test_img + 1) * 0.5
plt.figure(figsize=(20,20))

plt.subplot(131)
plt.imshow(cv2.cvtColor(test_img_disp, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off');

plt.subplot(132)
plt.imshow(style_img_disp)
plt.title('Style Image')
plt.axis('off');

plt.subplot(133)
plt.imshow(cv2.cvtColor(output_img_disp, cv2.COLOR_BGR2RGB));
plt.title('Transferred Image')
plt.axis('off');
```


![png](assets/example/style_transfer/fst_files/fst_25_0.png)

