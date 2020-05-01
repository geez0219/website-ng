# Advanced Tutorial 3: Operator

## Overview
In this tutorial, we will talk about:
* **Operator mechanism**
    * data
    * state
* **NumpyOp**
    * Delete
    * meta
    * Customization
* **TensorOp**
   * Customization

Here's one simple example of an operator. We will talk about in detail about how this Opearator works.


```python
from fastestimator.op.numpyop import NumpyOp

class AddOne(NumpyOp):
    def __init__(self, inputs, outputs, mode = None):
        super().__init__(inputs, outputs, mode)

    def forward(self, data, state):
        x, y = data
        x = x + 1
        y = y + 1
        return x, y

AddOneOp = AddOne(inputs=("x", "y"), outputs=("x_out", "y_out"))
```

## Operator Mechanism
We learned about the operator structure in [Tutorial 3](https://github.com/fastestimator/fastestimator/blob/master/tutorial/beginner/t03_operator.ipynb). Operators are used to build complex computation graphs in Fastestimator.

In Fastestimator, all the available data is held in a data dictionary during execution. An `Op` interacts with the required portion of this data using the keys specified through the `inputs` key, processes the data through the `forward` function and writes the values returned from the `forward` function to this data dictionary using the `outputs` key. An `Op` runs when it's `mode` matches the current execution mode. For more information on mode, you can go through [Tutorial 8](https://github.com/fastestimator/fastestimator/blob/master/tutorial/beginner/t08_mode.ipynb).

<img src="assets/tutorial/../resources/t03_advanced_operator_mechanism.png" alt="drawing" width="500"/>

### data
The data argument passes the portion of data dictionary corresponding to the keys passed as `inputs` to the forward function. If multiple keys are provided as inputs, data is a list of corresponding values of those keys.

### state
State stores meta information about training like mode, GradientTape for tensorflow etc. It is very unlikely that you would need to interact with it.

## NumpyOp
NumpyOp is used in pipeline for data pre-processing and augmentation. You can go through [Tutorial 4](https://github.com/fastestimator/fastestimator/blob/master/tutorial/beginner/t04_pipeline.ipynb) to get an overview of NumpyOp and their usage. Here, we will talk about some advanced NumpyOps.

### Delete
Delete op is used to delete keys from the data dictionary which are no longer required by the user. This helps in improving processing speed as we are holding only the required data in the memory. Below, we show it's usage.


```python
import fastestimator as fe
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop import Delete
from fastestimator.op.numpyop.meta import OneOf, Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, Rotate, VerticalFlip
from fastestimator.op.numpyop.univariate import Blur, Minmax, ChannelTranspose

train_data, eval_data = cifar10.load_data()

pipeline1 = fe.Pipeline(train_data=train_data,
                        eval_data=eval_data,
                        batch_size=4,
                        ops = [HorizontalFlip(image_in="x", image_out="x_mid", mode="train"),
                               Rotate(image_in="x_mid", image_out="x", mode="train", limit=45)])

pipeline2 = fe.Pipeline(train_data=train_data,
                        eval_data=eval_data,
                        batch_size=4,
                        ops = [HorizontalFlip(image_in="x", image_out="x_mid", mode="train"),
                               Rotate(image_in="x_mid", image_out="x", mode="train", limit=45),
                               Delete(keys="x_mid")])
```


```python
data1 = pipeline1.get_results()
print("Keys in pipeline: ", data1.keys())

data2 = pipeline2.get_results()
print("Keys in pipeline with Delete Op: ", data2.keys())
```

    Keys in pipeline:  dict_keys(['x', 'y', 'x_mid'])
    Keys in pipeline with Delete Op:  dict_keys(['x', 'y'])


### meta
meta ops are NumpyOps which operate on other NumpyOps. For example: `Sometimes` is a meta op which applies a given NumpyOp with the specified probability. `OneOf` applies only one randomly selected NumpyOp from the given list of NumpyOps.


```python
pipeline3 = fe.Pipeline(train_data=train_data,
                        eval_data=eval_data,
                        batch_size=4,
                        ops = [Sometimes(HorizontalFlip(image_in="x",
                                                        image_out="x_mid",
                                                        mode="train"), prob=0.5),
                               OneOf(Rotate(image_in="x_mid", image_out="x_out", mode="train", limit=45),
                                     VerticalFlip(image_in="x_mid", image_out="x_out", mode="train"),
                                     Blur(inputs="x_mid", outputs="x_out", mode="train", blur_limit=7))])
```

Plotting the results of the data pre-processing


```python
from matplotlib import pyplot as plt
import numpy as np

data3 = pipeline3.get_results()

for i in range(4):
    plt.subplot(131)
    plt.axis("off")
    plt.title("Input Image")
    plt.imshow(np.squeeze(data3["x"][i]))

    plt.subplot(132)
    plt.axis("off")
    plt.title("Sometimes Op")
    plt.imshow(np.squeeze(data3["x_mid"][i]))

    plt.subplot(133)
    plt.axis("off")
    plt.title("OneOf Op")
    plt.imshow(np.squeeze(data3["x_out"][i]))

    plt.show()
```


![png](assets/tutorial/t03_operator_files/t03_operator_14_0.png)



![png](assets/tutorial/t03_operator_files/t03_operator_14_1.png)



![png](assets/tutorial/t03_operator_files/t03_operator_14_2.png)



![png](assets/tutorial/t03_operator_files/t03_operator_14_3.png)


As you can see, Sometimes Op horizontally flips the image with 50% probability and OneOf applies, vertical flip, rotation and blur augmentations randomly.

### Customization
We can create a custom NumpyOp which suits our needs. Below, we showcase a custom NumpyOp which creates multiple random patches (crops) of images from each image.


```python
from albumentations.augmentations.transforms import RandomCrop

class Patch(NumpyOp):
    def __init__(self, height, width, inputs, outputs, mode = None, num_patch=2):
        super().__init__(inputs, outputs, mode)
        self.num_patch = num_patch
        self.crop_fn = RandomCrop(height=height, width=width, always_apply=True)

    def forward(self, data, state):
        image, label = data
        image = np.stack([self._gen_patch(image) for _ in range(self.num_patch)], axis=0)
        label = np.array([label for _ in range(self.num_patch)])
        return [image, label]

    def _gen_patch(self, data):
        data = self.crop_fn(image=data)
        return data["image"].astype(np.float32)
```

Let's create a pipeline and visualize the results.


```python
pipeline4 = fe.Pipeline(train_data=train_data,
                       eval_data=eval_data,
                       batch_size=8,
                       ops=[Minmax(inputs="x", outputs="x"),
                            Patch(height=24, width=24, inputs=["x", "y"], outputs=["x_out", "y_out"],
                                  num_patch=4)])
```


```python
data4 = pipeline4.get_results()
for i in range(4):
    plt.subplot(151)
    plt.title("Input Image")
    plt.axis("off")
    plt.imshow(np.squeeze(data4["x"][i]))

    plt.subplot(152)
    plt.axis("off")
    plt.title("Patch")
    plt.imshow(np.squeeze(data4["x_out"][i][0]))

    plt.subplot(153)
    plt.axis("off")
    plt.title("Patch")
    plt.imshow(np.squeeze(data4["x_out"][i][1]))

    plt.subplot(154)
    plt.axis("off")
    plt.title("Patch")
    plt.imshow(np.squeeze(data4["x_out"][i][2]))

    plt.subplot(155)
    plt.axis("off")
    plt.title("Patch")
    plt.imshow(np.squeeze(data4["x_out"][i][3]))

    plt.show()
```


![png](assets/tutorial/t03_operator_files/t03_operator_20_0.png)



![png](assets/tutorial/t03_operator_files/t03_operator_20_1.png)



![png](assets/tutorial/t03_operator_files/t03_operator_20_2.png)



![png](assets/tutorial/t03_operator_files/t03_operator_20_3.png)


## TensorOp
TensorOp is used to process tensor data. It's used in `Network` for graph-based operations. You can go through [Tutorial 6](https://github.com/fastestimator/fastestimator/blob/master/tutorial/beginner/t06_network.ipynb) to get an overview of TensorOp and their usage.

### Customization
We can create a custom TensorOp using TensorFlow or Pytorch library according to our requirements. Below, we showcase a custom TensorOp which reshapes the output of above Pipeline to make it compatible to the network.


```python
from fastestimator.op.tensorop import TensorOp
import tensorflow as tf

class DimensionAdjustment(TensorOp):
    def __init__(self, reduce_dim=[0, 1], inputs=None, outputs=None, mode=None):
        super().__init__(inputs, outputs, mode)
        self.reduce_dim = reduce_dim

    def forward(self, data, state):
        image, label = data
        image_out = tf.reshape(image, shape=self._new_shape(image))
        label_out = tf.reshape(label, shape=self._new_shape(label))
        return [image_out, label_out]

    def _new_shape(self, data):
        return [-1] + [data.shape[i] for i in range(len(data.shape)) if i not in self.reduce_dim]
```


```python
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp

pipeline5 = fe.Pipeline(train_data=train_data,
                       eval_data=eval_data,
                       batch_size=8,
                       ops=[Minmax(inputs="x", outputs="x"),
                            Patch(height=24, width=24, inputs=["x", "y"], outputs=["x", "y"],
                                  num_patch=4)])

model = fe.build(model_fn=lambda: LeNet(input_shape=(24, 24, 3)), optimizer_fn="adam")
network = fe.Network(ops=[
    DimensionAdjustment(reduce_dim=[0, 1], inputs=["x", "y"], outputs=["x", "y"]),
    ModelOp(model=model, inputs="x", outputs="y_pred"),
    CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
    UpdateOp(model=model, loss_name="ce")
])
```

Let's check the dimensions of Pipeline output and DimensionAdjustment TensorOp output.


```python
data5 = pipeline5.get_results()
result = network.transform(data5, mode="infer")

print("Pipeline Output, Image Shape: ", data5["x"].shape, " Label Shape: ", data5["y"].shape)
print("Result Image Shape: ", result["x"].shape, " Label Shape: ", result["y"].shape)
```

    Pipeline Output, Image Shape:  torch.Size([8, 4, 24, 24, 3])  Label Shape:  torch.Size([8, 4, 1])
    Result Image Shape:  (32, 24, 24, 3)  Label Shape:  (32, 1)

