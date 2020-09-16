# Advanced Tutorial 2: Pipeline

## Overview

In this tutorial, we will discuss the following topics:

* [Iterating Through Pipeline](./tutorials/advanced/t02_pipeline#ta02itp)
    * [Basic Concept](./tutorials/advanced/t02_pipeline#ta02bc)
    * [Example](./tutorials/advanced/t02_pipeline#ta02example)
* [Dropping Last Batch](./tutorials/advanced/t02_pipeline#ta02dlb)
* [Padding Batch Data](./tutorials/advanced/t02_pipeline#ta02pbd)
* [Benchmark Pipeline Speed](./tutorials/advanced/t02_pipeline#ta02bps)

In the [beginner tutorial 4](https://github.com/fastestimator/fastestimator/tree/master/tutorials/beginner/t04_pipeline), we learned how to build a data pipeline that handles data loading and preprocessing tasks efficiently. Now that you have understood some basic operations in the `Pipeline`, we will demonstrate some advanced concepts and how to leverage them to create efficient `Pipelines` in this tutorial.

<a id='ta02itp'></a>

## Iterating Through Pipeline

In many deep learning tasks, the parameters for preprocessing tasks are precomputed by looping through the dataset. For example, in the `ImageNet` dataset, people usually use a precomputed global pixel average for each channel to normalize the images. 

<a id='ta02bc'></a>

### Basic Concept

In this section, we will see how to iterate through the pipeline in FastEstimator. First we will create a sample NumpyDataset from the data dictionary and load it into a `Pipeline`:


```python
import numpy as np
from fastestimator.dataset.data import cifar10
    
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

    {'x': tensor([[0.1288, 0.2118],
            [0.9344, 0.5583],
            [0.0879, 0.5939]], dtype=torch.float64), 'y': tensor([[0.8071],
            [0.8469],
            [0.9160]], dtype=torch.float64)}
    {'x': tensor([[0.7866, 0.8248],
            [0.3285, 0.9311],
            [0.7637, 0.9474]], dtype=torch.float64), 'y': tensor([[0.5504],
            [0.8430],
            [0.7415]], dtype=torch.float64)}
    {'x': tensor([[0.3689, 0.3373],
            [0.3407, 0.0571],
            [0.2216, 0.1906]], dtype=torch.float64), 'y': tensor([[0.6517],
            [0.4824],
            [0.5171]], dtype=torch.float64)}
    {'x': tensor([[0.6018, 0.4306]], dtype=torch.float64), 'y': tensor([[0.0023]], dtype=torch.float64)}


<a id='ta02example'></a>

### Example

Let's say we have CIFAR-10 dataset and we want to find global average pixel value over three channels:


```python
from fastestimator.dataset.data import cifar10

cifar_train, _ = cifar10.load_data()
```

We will take the `batch_size` 64 and load the data into `Pipeline`


```python
pipeline_cifar = fe.Pipeline(train_data=cifar_train, batch_size=64)
```

Now we will iterate through batch data and compute the mean pixel values for all three channels of the dataset. 


```python
loader_fe = pipeline_cifar.get_loader(mode="train", shuffle=False)
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

For illustration, we will create a `Pipeline` for the CIFAR-10 dataset with list of Numpy operators that expand dimensions, apply `Minmax` and finally `Rotate` the input images: 


```python
from fastestimator.op.numpyop.univariate import Minmax, ExpandDims
from fastestimator.op.numpyop.multivariate import Rotate

pipeline = fe.Pipeline(train_data=cifar_train,
                       ops=[ExpandDims(inputs="x", outputs="x"),
                            Minmax(inputs="x", outputs="x_out"),
                            Rotate(image_in="x_out", image_out="x_out", limit=180)],
                      batch_size=64)
```

Let's benchmark the pre-processing speed for this pipeline in training mode:


```python
pipeline_cifar.benchmark(mode="train")
```

    FastEstimator: Step: 100, Epoch: 1, Steps/sec: 797.9008250605733
    FastEstimator: Step: 200, Epoch: 1, Steps/sec: 2249.3393577839283
    FastEstimator: Step: 300, Epoch: 1, Steps/sec: 2236.913774803168
    FastEstimator: Step: 400, Epoch: 1, Steps/sec: 2244.6406454903963
    FastEstimator: Step: 500, Epoch: 1, Steps/sec: 2303.2515324338206
    FastEstimator: Step: 600, Epoch: 1, Steps/sec: 2250.139806811566
    FastEstimator: Step: 700, Epoch: 1, Steps/sec: 2310.7264336983017

