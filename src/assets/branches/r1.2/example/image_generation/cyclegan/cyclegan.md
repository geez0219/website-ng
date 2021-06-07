# Horse to Zebra Unpaired Image Translation with CycleGAN in FastEstimator

This notebook demonstrates how to perform an unpaired image to image translation using CycleGAN in FastEstimator.
The details of the method is found in [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).
We will specifically look at the problem of translating horse images to zebra images.


```python
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_

import fastestimator as fe
from fastestimator.backend import reduce_mean
```


```python
#Parameters
epochs = 200
batch_size = 1
max_train_steps_per_epoch = None
save_dir=tempfile.mkdtemp()
weight = 10.0
data_dir=None
```

## Building Components

### Downloading the data
First, we will download the dataset of horses and zebras via our dataset API.
The images will be first downloaded from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).
As this task requires an unpaired datasets of horse and zebra images, horse2zebra dataset is implemented using `BatchDataset` in FastEstimator. Hence, we need to specify the batch size while loading the dataset.


```python
from fastestimator.dataset.data.horse2zebra import load_data
train_data, test_data = load_data(batch_size=batch_size, root_dir=data_dir)
```

### Step 1: Create pipeline

Let's create the pipeline. As, batch_size must be `None` when BatchDataset is being used, we will not provide the batch_size argument.


```python
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, RandomCrop, Resize
from fastestimator.op.numpyop.univariate import ChannelTranspose, Normalize, ReadImage

pipeline = fe.Pipeline(
    train_data=train_data,
    ops=[
        ReadImage(inputs=["A", "B"], outputs=["A", "B"]),
        Normalize(inputs=["A", "B"], outputs=["real_A", "real_B"], mean=1.0, std=1.0, max_pixel_value=127.5),
        Resize(height=286, width=286, image_in="real_A", image_out="real_A", mode="train"),
        RandomCrop(height=256, width=256, image_in="real_A", image_out="real_A", mode="train"),
        Resize(height=286, width=286, image_in="real_B", image_out="real_B", mode="train"),
        RandomCrop(height=256, width=256, image_in="real_B", image_out="real_B", mode="train"),
        Sometimes(HorizontalFlip(image_in="real_A", image_out="real_A", mode="train")),
        Sometimes(HorizontalFlip(image_in="real_B", image_out="real_B", mode="train")),
        ChannelTranspose(inputs=["real_A", "real_B"], outputs=["real_A", "real_B"])
    ])
```

We can visualize sample images from the `pipeline` using `get_results` method.


```python
def Minmax(img):
    img_max = np.max(img)
    img_min = np.min(img)
    img = (img - img_min)/max((img_max - img_min), 1e-7)
    img = (img*255).astype(np.uint8)
    return img
```


```python
sample_batch = pipeline.get_results()
horse_img = sample_batch["real_A"][0]
horse_img = np.transpose(horse_img.numpy(), (1, 2, 0))
horse_img = np.expand_dims(Minmax(horse_img), 0)

zebra_img = sample_batch["real_B"][0]
zebra_img = np.transpose(zebra_img.numpy(), (1, 2, 0))
zebra_img = np.expand_dims(Minmax(zebra_img), 0)

img = fe.util.ImgData(Sample_Horse_Image=horse_img, Sample_Zebra_Image=zebra_img)
fig = img.paint_figure()
```


    
![png](assets/branches/r1.2/example/image_generation/cyclegan/cyclegan_files/cyclegan_11_0.png)
    


### Step 2: Create Network

In CycleGAN, there are 2 generators and 2 discriminators being trained.
* Generator `g_AtoB` learns to map horse images to zebra images
* Generator `g_BtoA` learns to map zebra images to horse images
* Discriminator `d_A` learns to differentiate between real hores images and fake horse images produced by `g_BtoA`
* Discriminator `d_B` learns to differentiate between image zebra and fake zebra images produced by `g_AtoB`

The architecture of generator is a modified resnet, and the architecture of discriminator is a PatchGAN.


```python
class ResidualBlock(nn.Module):
    """Residual block architecture"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.layers = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size),
                                    nn.InstanceNorm2d(out_channels))

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                normal_(layer.weight.data, mean=0, std=0.02)

    def forward(self, x):
        x_out = self.layers(x)
        x_out = x_out + x
        return x_out


class Discriminator(nn.Module):
    """Discriminator network architecture"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                    nn.InstanceNorm2d(128),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                                    nn.InstanceNorm2d(256),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(256, 512, kernel_size=4, stride=1),
                                    nn.InstanceNorm2d(512),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 1, kernel_size=4, stride=1))

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                normal_(layer.weight.data, mean=0, std=0.02)

    def forward(self, x):
        x = self.layers(x)
        return x


class Generator(nn.Module):
    """Generator network architecture"""
    def __init__(self, num_blocks=9):
        super().__init__()
        self.layers1 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(3, 64, kernel_size=7, stride=1),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                     nn.InstanceNorm2d(256),
                                     nn.ReLU(inplace=True))
        self.resblocks = nn.Sequential(*[ResidualBlock(256, 256) for i in range(num_blocks)])
        self.layers2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(3),
                                     nn.Conv2d(64, 3, kernel_size=7, stride=1))

        for block in [self.layers1, self.layers2]:
            for layer in block:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                    normal_(layer.weight.data, mean=0, std=0.02)

    def forward(self, x):
        x = self.layers1(x)
        x = self.resblocks(x)
        x = self.layers2(x)
        x = torch.tanh(x)
        return x
```


```python
g_AtoB = fe.build(model_fn=Generator,
                  model_name="g_AtoB",
                  optimizer_fn=lambda x: torch.optim.Adam(x, lr=2e-4, betas=(0.5, 0.999)))
g_BtoA = fe.build(model_fn=Generator,
                  model_name="g_BtoA",
                  optimizer_fn=lambda x: torch.optim.Adam(x, lr=2e-4, betas=(0.5, 0.999)))
d_A = fe.build(model_fn=Discriminator,
               model_name="d_A",
               optimizer_fn=lambda x: torch.optim.Adam(x, lr=2e-4, betas=(0.5, 0.999)))
d_B = fe.build(model_fn=Discriminator,
               model_name="d_B",
               optimizer_fn=lambda x: torch.optim.Adam(x, lr=2e-4, betas=(0.5, 0.999)))
```

### Defining Loss functions
For each network, we need to define associated losses.

Because horse images and zebra images are unpaired, the loss of generator is quite complex. 
The generator's loss is composed of three terms: 
    * adversarial
    * cycle-consistency
    * identity.
The cycle-consistency term and identity term are weighted by a parameter `LAMBDA`. In the paper the authors used 10 for `LAMBDA`.

Let's consider computing the loss for `g_AtoB` which translates horses to zebras.
1. Adversarial term that is computed as binary cross entropy between ones and `d_A`'s prediction on the translated images
2. Cycle consistency term is computed with mean absolute error between original *horse* images and the cycled horse images that are translated *forward* by `g_AtoB` and then *backward* by `g_BtoA`.
3. Identity term that is computed with the mean absolute error between original *zebra* and the output of `g_AtoB` on these images.

The discriminator's loss is the standard adversarial loss that is computed as binary cross entropy between:
* Ones and real images
* Zeros and fake images


```python
from fastestimator.op.tensorop import TensorOp

class GLoss(TensorOp):
    """TensorOp to compute generator loss"""
    def __init__(self, inputs, weight, device, outputs=None, mode=None, average_loss=True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_fn = nn.MSELoss(reduction="none")
        self.LAMBDA = weight
        self.device = device
        self.average_loss = average_loss

    def _adversarial_loss(self, fake_img):
        return torch.mean(self.loss_fn(fake_img, torch.ones_like(fake_img, device=self.device)), dim=(2, 3))

    def _identity_loss(self, real_img, same_img):
        return 0.5 * self.LAMBDA * torch.mean(torch.abs(real_img - same_img), dim=(1, 2, 3))

    def _cycle_loss(self, real_img, cycled_img):
        return self.LAMBDA * torch.mean(torch.abs(real_img - cycled_img), dim=(1, 2, 3))

    def forward(self, data, state):
        real_img, fake_img, cycled_img, same_img = data
        total_loss = self._adversarial_loss(fake_img) + self._identity_loss(real_img, same_img) + self._cycle_loss(
            real_img, cycled_img)

        if self.average_loss:
            total_loss = reduce_mean(total_loss)

        return total_loss


class DLoss(TensorOp):
    """TensorOp to compute discriminator loss"""
    def __init__(self, inputs, device, outputs=None, mode=None, average_loss=True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_fn = nn.MSELoss(reduction="none")
        self.device = device
        self.average_loss = average_loss

    def forward(self, data, state):
        real_img, fake_img = data
        real_img_loss = torch.mean(self.loss_fn(real_img, torch.ones_like(real_img, device=self.device)), dim=(2, 3))
        fake_img_loss = torch.mean(self.loss_fn(fake_img, torch.zeros_like(real_img, device=self.device)), dim=(2, 3))
        total_loss = real_img_loss + fake_img_loss

        if self.average_loss:
            total_loss = reduce_mean(total_loss)

        return 0.5 * total_loss
```

We implement an image buffer as a `TensorOp` which stores the previous images produced by the generators to updated the discriminators as outlined in [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).


```python
class Buffer(TensorOp):
    def __init__(self, image_in=None, image_out=None, mode=None, buffer_size=50):
        super().__init__(inputs=image_in, outputs=image_out, mode=mode)
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.image_buffer = []

    def forward(self, data, state):
        output = []
        for image in data:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.image_buffer.append(image)
                output.append(image)
                self.num_imgs += 1
            else:
                if np.random.uniform() > 0.5:
                    idx = np.random.randint(self.buffer_size)
                    temp = self.image_buffer[idx].clone()
                    self.image_buffer[idx] = image
                    output.append(temp)
                else:
                    output.append(image)

        output = torch.cat(output, 0)
        return output
```

Once associated losses are defined, we can now define the `Network` object.


```python
from fastestimator.op.tensorop.model import ModelOp, UpdateOp

device = "cuda" if torch.cuda.is_available() else "cpu"

network = fe.Network(ops=[
    ModelOp(inputs="real_A", model=g_AtoB, outputs="fake_B"),
    ModelOp(inputs="real_B", model=g_BtoA, outputs="fake_A"),
    Buffer(image_in="fake_A", image_out="buffer_fake_A"),
    Buffer(image_in="fake_B", image_out="buffer_fake_B"),
    ModelOp(inputs="real_A", model=d_A, outputs="d_real_A"),
    ModelOp(inputs="fake_A", model=d_A, outputs="d_fake_A"),
    ModelOp(inputs="buffer_fake_A", model=d_A, outputs="buffer_d_fake_A"),
    ModelOp(inputs="real_B", model=d_B, outputs="d_real_B"),
    ModelOp(inputs="fake_B", model=d_B, outputs="d_fake_B"),
    ModelOp(inputs="buffer_fake_B", model=d_B, outputs="buffer_d_fake_B"),
    ModelOp(inputs="real_A", model=g_BtoA, outputs="same_A"),
    ModelOp(inputs="fake_B", model=g_BtoA, outputs="cycled_A"),
    ModelOp(inputs="real_B", model=g_AtoB, outputs="same_B"),
    ModelOp(inputs="fake_A", model=g_AtoB, outputs="cycled_B"),
    GLoss(inputs=("real_A", "d_fake_B", "cycled_A", "same_A"), weight=weight, device=device, outputs="g_AtoB_loss"),
    GLoss(inputs=("real_B", "d_fake_A", "cycled_B", "same_B"), weight=weight, device=device, outputs="g_BtoA_loss"),
    DLoss(inputs=("d_real_A", "buffer_d_fake_A"), outputs="d_A_loss", device=device),
    DLoss(inputs=("d_real_B", "buffer_d_fake_B"), outputs="d_B_loss", device=device),
    UpdateOp(model=g_AtoB, loss_name="g_AtoB_loss"),
    UpdateOp(model=g_BtoA, loss_name="g_BtoA_loss"),
    UpdateOp(model=d_A, loss_name="d_A_loss"),
    UpdateOp(model=d_B, loss_name="d_B_loss")
])
```

Here, we use a linear learning rate decay for training. 


```python
def lr_schedule(epoch):
    if epoch<=100:
        lr = 2e-4
    else:
        lr = 2e-4*(200 - epoch)/100
    return lr
```

In this example we will use `ModelSaver` traces to save the two generators `g_AtoB` and `g_BtoA` throughout training and `LRScheduler` traces to update the learning rate.


```python
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import ModelSaver

traces = [
    ModelSaver(model=g_AtoB, save_dir=save_dir, frequency=10),
    ModelSaver(model=g_BtoA, save_dir=save_dir, frequency=10),
    LRScheduler(model=g_AtoB, lr_fn=lr_schedule),
    LRScheduler(model=g_BtoA, lr_fn=lr_schedule),
    LRScheduler(model=d_A, lr_fn=lr_schedule),
    LRScheduler(model=d_B, lr_fn=lr_schedule)
]
```

### Step 3: Estimator
Finally, we are ready to define `Estimator` object and then call `fit` method to start the training.
Just for the sake of demo purpose, we would only run 50 epochs.


```python
estimator = fe.Estimator(network=network, 
                         pipeline=pipeline, 
                         epochs=epochs, 
                         traces=traces,
                         log_steps=1000, 
                         max_train_steps_per_epoch=max_train_steps_per_epoch)
```

## Training


```python
estimator.fit()
```

## Inferencing
Below are infering results of the two generators. 


```python
idx = np.random.randint(len(test_data))
data = test_data[idx][0]
result = pipeline.transform(data, mode="infer")
```


```python
network = fe.Network(ops=[
    ModelOp(inputs="real_A", model=g_AtoB, outputs="fake_B"),
    ModelOp(inputs="real_B", model=g_BtoA, outputs="fake_A"),
])

predictions = network.transform(result, mode="infer")
horse_img = np.transpose(predictions["real_A"].numpy(), (0, 2, 3, 1))
zebra_img = np.transpose(predictions["real_B"].numpy(), (0, 2, 3, 1))
fake_zebra = np.transpose(predictions["fake_B"].numpy(), (0, 2, 3, 1))
fake_horse = np.transpose(predictions["fake_A"].numpy(), (0, 2, 3, 1))
```


```python
img1 = fe.util.ImgData(Real_Horse_Image=horse_img, Translated_Zebra_Image=fake_zebra)
fig1 = img1.paint_figure()

img2 = fe.util.ImgData(Real_Zebra_Image=zebra_img, Translated_Horse_Image=fake_horse)
fig2 = img2.paint_figure()
```


    
![png](assets/branches/r1.2/example/image_generation/cyclegan/cyclegan_files/cyclegan_34_0.png)
    



    
![png](assets/branches/r1.2/example/image_generation/cyclegan/cyclegan_files/cyclegan_34_1.png)
    


Note the addition of zebra-like stripe texture on top of horses when translating from horses to zebras.
When translating zebras to horses, we can observe that generator removes the stripe texture from zebras.
