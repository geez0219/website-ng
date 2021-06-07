# Advanced Tutorial 2: Pipeline

## Overview

In this tutorial, we will discuss the following topics:

* [Iterating Through Pipeline](tutorials/r1.2/advanced/t02_pipeline/#ta02itp)
    * [Basic Concept](tutorials/r1.2/advanced/t02_pipeline/#ta02bc)
    * [Example](tutorials/r1.2/advanced/t02_pipeline/#ta02example)
* [Dropping Last Batch](tutorials/r1.2/advanced/t02_pipeline/#ta02dlb)
* [Padding Batch Data](tutorials/r1.2/advanced/t02_pipeline/#ta02pbd)
* [Benchmark Pipeline Speed](tutorials/r1.2/advanced/t02_pipeline/#ta02bps)

In the [Beginner Tutorial 4](tutorials/r1.2/beginner/t04_pipeline), we learned how to build a data pipeline that handles data loading and preprocessing tasks efficiently. Now that you have understood some basic operations in the `Pipeline`, we will demonstrate some advanced concepts and how to leverage them to create efficient `Pipelines` in this tutorial.

<a id='ta02itp'></a>

## Iterating Through Pipeline

In many deep learning tasks, the parameters for preprocessing tasks are precomputed by looping through the dataset. For example, in the `ImageNet` dataset, people usually use a precomputed global pixel average for each channel to normalize the images. 

<a id='ta02bc'></a>

### Basic Concept

In this section, we will see how to iterate through the pipeline in FastEstimator. First we will create a sample NumpyDataset from the data dictionary and load it into a `Pipeline`:


```python
import numpy as np
from fastestimator.dataset.data import cifair10
    
# sample numpy array to later create datasets from them
x_train, y_train = (np.random.sample((10, 2)), np.random.sample((10, 1)))
train_data = {"x": x_train, "y": y_train}
```


```python
import fastestimator as fe
from fastestimator.dataset.numpy_dataset import NumpyDataset

# create NumpyDataset from the sample data
dataset_fe = NumpyDataset(train_data)

pipeline_fe = fe.Pipeline(train_data=dataset_fe, batch_size=3)
```

Let's get the loader object for the `Pipeline`, then iterate through the loader with a for loop:


```python
loader_fe = pipeline_fe.get_loader(mode="train")

for batch in loader_fe:
    print(batch)
```

    {'x': tensor([[0.5085, 0.0519],
            [0.8167, 0.1808],
            [0.9175, 0.5209]], dtype=torch.float64), 'y': tensor([[0.5407],
            [0.7994],
            [0.4728]], dtype=torch.float64)}
    {'x': tensor([[0.9302, 0.6404],
            [0.1795, 0.1212],
            [0.8716, 0.9381]], dtype=torch.float64), 'y': tensor([[0.4747],
            [0.4103],
            [0.3916]], dtype=torch.float64)}
    {'x': tensor([[0.9929, 0.9415],
            [0.6404, 0.8039],
            [0.1624, 0.9285]], dtype=torch.float64), 'y': tensor([[0.1118],
            [0.8162],
            [0.7057]], dtype=torch.float64)}
    {'x': tensor([[0.0748, 0.8554]], dtype=torch.float64), 'y': tensor([[0.2276]], dtype=torch.float64)}


<a id='ta02example'></a>

### Example

Let's say we have the ciFAIR-10 dataset and we want to find global average pixel value over three channels:


```python
from fastestimator.dataset.data import cifair10

cifair_train, _ = cifair10.load_data()
```

We will take the `batch_size` 64 and load the data into `Pipeline`


```python
pipeline_cifair = fe.Pipeline(train_data=cifair_train, batch_size=64)
```

Now we will iterate through batch data and compute the mean pixel values for all three channels of the dataset. 


```python
loader_fe = pipeline_cifair.get_loader(mode="train", shuffle=False)
mean_arr = np.zeros((3))
for i, batch in enumerate(loader_fe):
    mean_arr = mean_arr + np.mean(batch["x"].numpy(), axis=(0, 1, 2))
mean_arr = mean_arr / (i+1)
```


```python
print("Mean pixel value over the channels are: ", mean_arr)
```

    Mean pixel value over the channels are:  [125.32287898 122.96682199 113.8856495 ]


<a id='ta02dlb'></a>

## Dropping Last Batch

If the total number of dataset elements is not divisible by the `batch_size`, by default, the last batch will have less data than other batches.  To drop the last batch we can set `drop_last` to `True`. Therefore, if the last batch is incomplete it will be dropped.


```python
pipeline_fe = fe.Pipeline(train_data=dataset_fe, batch_size=3, drop_last=True)
```

<a id='ta02pbd'></a>

## Padding Batch Data

There might be scenario where the input tensors have different dimensions within a batch. For example, in Natural Language Processing, we have input strings with different lengths. For that we need to pad the data to the maximum length within the batch.


To further illustrate in code, we will take numpy array that contains different shapes of array elements and load it into the `Pipeline`.


```python
# define numpy arrays with different shapes
elem1 = np.array([4, 5])
elem2 = np.array([1, 2, 6])
elem3 = np.array([3])

# create train dataset
x_train = np.array([elem1, elem2, elem3])
train_data = {"x": x_train}
dataset_fe = NumpyDataset(train_data)
```

We will set any `pad_value` that we want to append at the end of the tensor data. `pad_value` can be either `int` or `float`:


```python
pipeline_fe = fe.Pipeline(train_data=dataset_fe, batch_size=3, pad_value=0)
```

Now let's print the batch data after padding:


```python
for elem in iter(pipeline_fe.get_loader(mode='train', shuffle=False)):
    print(elem)
```

    {'x': tensor([[4, 5, 0],
            [1, 2, 6],
            [3, 0, 0]])}


<a id='ta02bps'></a>

## Benchmark Pipeline Speed

It is often the case that the bottleneck of deep learning training is the data pipeline. As a result, the GPU may be underutilized. FastEstimator provides a method to check the speed of a `Pipeline` in order to help diagnose any potential problems. The way to benchmark `Pipeline` speed in FastEstimator is very simple: call `Pipeline.benchmark`.

For illustration, we will create a `Pipeline` for the ciFAIR-10 dataset with list of Numpy operators that expand dimensions, apply `Minmax` and finally `Rotate` the input images: 


```python
from fastestimator.op.numpyop.univariate import Minmax, ExpandDims
from fastestimator.op.numpyop.multivariate import Rotate

pipeline = fe.Pipeline(train_data=cifair_train,
                       ops=[Minmax(inputs="x", outputs="x_out"),
                            Rotate(image_in="x_out", image_out="x_out", limit=180),
                            ExpandDims(inputs="x_out", outputs="x_out", mode="train")],
                       batch_size=64)
```

Let's benchmark the pre-processing speed for this pipeline in training mode:


```python
pipeline.benchmark(mode="train")
```

    FastEstimator: Step: 100, Epoch: 1, Steps/sec: 355.0314672561461
    FastEstimator: Step: 200, Epoch: 1, Steps/sec: 688.959209809261
    FastEstimator: Step: 300, Epoch: 1, Steps/sec: 646.3625572114369
    FastEstimator: Step: 400, Epoch: 1, Steps/sec: 709.4726870671318
    FastEstimator: Step: 500, Epoch: 1, Steps/sec: 654.4829388343634
    FastEstimator: Step: 600, Epoch: 1, Steps/sec: 716.1617101086067
    FastEstimator: Step: 700, Epoch: 1, Steps/sec: 635.2802801079024
    
    Breakdown of time taken by Pipeline Operations (train epoch 1)
    Op         : Inputs : Outputs :  Time
    --------------------------------------
    Minmax     : x      : x_out   : 39.42%
    Rotate     : x_out  : x_out   : 49.45%
    ExpandDims : x_out  : x_out   : 11.13%

