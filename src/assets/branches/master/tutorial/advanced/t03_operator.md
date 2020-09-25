# Advanced Tutorial 3: Operator

## Overview
In this tutorial, we will discuss:
* [Operator Mechanism](./tutorials/master/advanced/t03_operator#ta03om)
    * [data](./tutorials/master/advanced/t03_operator#ta03data)
    * [state](./tutorials/master/advanced/t03_operator#ta03state)
* [NumpyOp](./tutorials/master/advanced/t03_operator#ta03no)
    * [DeleteOp](./tutorials/master/advanced/t03_operator#ta03do)
    * [MetaOp](./tutorials/master/advanced/t03_operator#ta03mo)
    * [Customizing NumpyOps](./tutorials/master/advanced/t03_operator#ta03cn)
* [TensorOp](./tutorials/master/advanced/t03_operator#ta03to)
   * [Customizing TensorOps](./tutorials/master/advanced/t03_operator#ta03ct)
* [Related Apphub Examples](./tutorials/master/advanced/t03_operator#ta03rae)

<a id='ta03om'></a>

## Operator Mechanism
We learned about the operator structure in [Beginner tutorial 3](./tutorials/master/beginner/t03_operator). Operators are used to build complex computation graphs in FastEstimator.

In FastEstimator, all the available data is held in a data dictionary during execution. An `Op` runs when it's `mode` matches the current execution mode. For more information on mode, you can go through [Beginner tutorial 8](./tutorials/master/beginner/t08_mode).

Here's one simple example of an operator:


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

An `Op` interacts with the required portion of this data using the keys specified through the `inputs` key, processes the data through the `forward` function and writes the values returned from the `forward` function to this data dictionary using the `outputs` key. The processes are illustrated in the diagram below:

<img src="assets/branches/master/tutorial/../resources/t03_advanced_operator_mechanism.png" alt="drawing" width="500"/>

<a id='ta03data'></a>

### data
The data argument in the `forward` function passes the portion of data dictionary corresponding to the Operator's `inputs` into the forward function. If multiple keys are provided as `inputs`, the data will be a list of corresponding to the values of those keys. 

<a id='ta03state'></a>

### state
The state argument in the `forward` function stores meta information about training like the current mode, GradientTape for tensorflow, etc. It is very unlikely that you would need to interact with it.

<a id='ta03no'></a>

## NumpyOp
NumpyOp is used in `Pipeline` for data pre-processing and augmentation. You can go through [Beginner tutorial 4](./tutorials/master/beginner/t04_pipeline) to get an overview of NumpyOp and their usage. Here, we will talk about some advanced NumpyOps.

<a id='ta03do'></a>

### DeleteOp
Delete op is used to delete keys from the data dictionary which are no longer required by the user. This helps in improving processing speed as we are holding only the required data in the memory. Let's see its usage:


```python
import fastestimator as fe
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop import Delete, LambdaOp
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


<a id='ta03mo'></a>

### MetaOp
Meta ops are NumpyOps which operate on other NumpyOps. For example: `Sometimes` is a meta op which applies a given NumpyOp with the specified probability. `OneOf` applies only one randomly selected NumpyOp from the given list of NumpyOps.   


```python
pipeline3 = fe.Pipeline(train_data=train_data,
                        eval_data=eval_data,
                        batch_size=4,
                        ops = [LambdaOp(fn=lambda x: x, inputs="x", outputs="x_mid"), #create intermediate key
                               Sometimes(HorizontalFlip(image_in="x", 
                                                        image_out="x_mid",
                                                        mode="train"), prob=0.5), 
                               OneOf(Rotate(image_in="x_mid", image_out="x_out", mode="train", limit=45), 
                                     VerticalFlip(image_in="x_mid", image_out="x_out", mode="train"), 
                                     Blur(inputs="x_mid", outputs="x_out", mode="train", blur_limit=7))])
```

Plotting the results of the data pre-processing


```python
from fastestimator.util import to_number

data3 = pipeline3.get_results()
img = fe.util.ImgData(Input_Image=to_number(data3["x"]), Sometimes_Op=to_number(data3["x_mid"]), OneOf_Op=to_number(data3["x_out"]))
fig = img.paint_figure()
```


![png](assets/branches/master/tutorial/advanced/t03_operator_files/t03_operator_21_0.png)


As you can see, Sometimes Op horizontally flips the image with 50% probability and OneOf applies either a vertical flip, rotation, or blur augmentation randomly.

<a id='ta03cn'></a>

### Customizing NumpyOps
We can create a custom NumpyOp which suits our needs. Below, we showcase a custom NumpyOp which creates multiple random patches (crops) of images from each image.


```python
from albumentations.augmentations.transforms import RandomCrop
import numpy as np

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
from fastestimator.util import to_number

data4 = pipeline4.get_results()
img = fe.util.ImgData(Input_Image=to_number(data4["x"]), 
                      Patch_0=to_number(data4["x_out"])[:,0,:,:,:], 
                      Patch_1=to_number(data4["x_out"])[:,1,:,:,:], 
                      Patch_2=to_number(data4["x_out"])[:,2,:,:,:], 
                      Patch_3=to_number(data4["x_out"])[:,3,:,:,:])
fig = img.paint_figure()
```


![png](assets/branches/master/tutorial/advanced/t03_operator_files/t03_operator_28_0.png)


<a id='ta03to'></a>

## TensorOp
`TensorOps` are used to process tensor data. They are used within a `Network` for graph-based operations. You can go through [Beginner tutorial 6](./tutorials/master/beginner/t06_network) to get an overview of `TensorOps` and their usages.

<a id='ta03ct'></a>

### Customizing TensorOps
We can create a custom `TensorOp` using TensorFlow or Pytorch library calls according to our requirements. Below, we showcase a custom `TensorOp` which combines the batch dimension and patch dimension from the output of the above `Pipeline` to make it compatible to the `Network`.  


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

Let's check the dimensions the of Pipeline output and DimensionAdjustment TensorOp output.


```python
data5 = pipeline5.get_results()
result = network.transform(data5, mode="infer")

print(f"Pipeline Output, Image Shape: {data5['x'].shape}, Label Shape: {data5['y'].shape}")
print(f"Result Image Shape: {result['x'].shape}, Label Shape: {result['y'].shape}")
```

    Pipeline Output, Image Shape: torch.Size([8, 4, 24, 24, 3]), Label Shape: torch.Size([8, 4, 1])
    Result Image Shape: (32, 24, 24, 3), Label Shape: (32, 1)


<a id='ta03rae'></a>

## Apphub Examples 

You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [Fast Style Transfer](./examples/master/style_transfer/fst)
* [Convolutional Variational AutoEncoder](./examples/master/image_generation/cvae)
* [Semantic Segmentation](./examples/master/semantic_segmentation/unet)
