<h1>Advanced Tutorial 2: Pipeline</h1>

<h2>Overview</h2>

In the beginner's tutorial of `Pipeline`, we learned how to build data pipeline that handles data loading and preprocessing tasks efficiently. Now that you have understood some basic operations in the `Pipeline`, we will demonstrate some advanced concepts and how to leverage them to create efficient `Pipeline` in this tutorial.

In this tutorial we will discuss following topics,

* How to iterate through the pipeline data
    * Basic concept
    * Example use case
* Dropping the last batch
* Handling the batch padding
* How to benchmark Pipeline performance

<h2>How to iterate through the pipeline data</h2>

We will first see how to iterate through the pipeline batch data. For example if we want to calculate global mean of pixel value or standard deviation over the channels we could iterate through the batch data and compute them.

First we will create sample NumpyDataset from the data dictionary and load it into `Pipeline`.


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

Let's get the loader object for the `Pipeline` we defined and iterate over the dataset that was loaded into the dataloader.


```python
loader_fe = pipeline_fe.get_loader(mode="train")

for batch in loader_fe:
    print(batch)
```

    {'x': tensor([[0.4575, 0.8058],
            [0.8462, 0.1682],
            [0.1016, 0.3228]], dtype=torch.float64), 'y': tensor([[0.8675],
            [0.0056],
            [0.3656]], dtype=torch.float64)}
    {'x': tensor([[0.6502, 0.7932],
            [0.5179, 0.5414],
            [0.9607, 0.0284]], dtype=torch.float64), 'y': tensor([[0.6766],
            [0.4403],
            [0.3337]], dtype=torch.float64)}
    {'x': tensor([[0.5675, 0.8176],
            [0.9654, 0.8325],
            [0.0961, 0.1680]], dtype=torch.float64), 'y': tensor([[0.8057],
            [0.9169],
            [0.2998]], dtype=torch.float64)}
    {'x': tensor([[0.8078, 0.3384]], dtype=torch.float64), 'y': tensor([[0.2901]], dtype=torch.float64)}


<h3>Example use case</h3>

Let's say we have CIFAR-10 dataset and we want to find global average pixel value over three channels then we can loop through the batch data and quickly compute the value.


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
print("Mean pixel value over the channels: ", mean_arr)
```

    Mean pixel value over the channels:  [125.32287898 122.96682199 113.8856495 ]


<h2>Dropping the last batch</h2>

When we specify `batch_size` in the `Pipeline`, it will combine consecutive number of tensors into a batch and resulting shape will be <br><b>batch_size * shape of input tensor</b><br> However, if `batch_size` does not divide the input data evenly then last batch could have different batch_size than other batches.<br>
To drop the last batch we can set `drop_last` to `True`. Therefore, if the last batch is incomplete it will be dropped.


```python
pipeline_fe = fe.Pipeline(train_data=dataset_fe, batch_size=3, drop_last=True)
```

<h2>Handling the batch padding</h2>

In the previous section we saw that if last batch has different shape than rest of the batches then we can drop the last batch. But there might be scenario where the input tensors that are batched have different dimensions i.e. In Natural language processing problems we can have input strings can have different lengths. For that the tensors are padded out to the maximum length of the all the tensors in the dataset.


We will take numpy array that contains different shapes of array elements and load it into the `Pipeline`.


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

We will set any `pad_value` that we want to append at the end of the tensor data. `pad_value` must be either `int` or `float`


```python
pipeline_fe = fe.Pipeline(train_data=dataset_fe, batch_size=3, pad_value=0)
```

Now let's iterate over the batch data


```python
for elem in iter(pipeline_fe.get_loader(mode='train', shuffle=False)):
    print(elem)
```

    {'x': tensor([[4, 5, 0],
            [1, 2, 6],
            [3, 0, 0]])}


<h2>Benchmarking pipeline performance</h2>

In the ideal world, deep learning scientists would need to evaluate costs and speed in either in terms of data processing or model training before deploying. That makes benchmarking such tasks significant as we need good summary of the measures.<br>
`Pipeline.benchmark` provides that important feature of benchmarking processing speed of pre-processing operations in the `Pipeline`

We will create `Pipeline` for the CIFAR-10 dataset with list of numpy operators that expand dimensions, apply minmax scaler and finally rotate the input images. 


```python
from fastestimator.op.numpyop.univariate import Minmax, ExpandDims
from fastestimator.op.numpyop.multivariate import Rotate

pipeline = fe.Pipeline(train_data=cifar_train,
                       ops=[ExpandDims(inputs="x", outputs="x"),
                            Minmax(inputs="x", outputs="x_out"),
                            Rotate(image_in="x_out", image_out="x_out", limit=180)],
                      batch_size=64)
```

Let's benchmark the processing speed in the training mode.


```python
pipeline_cifar.benchmark(mode="train")
```

    FastEstimator: Step: 100, Epoch: 1, Steps/sec: 306.3574085435541
    FastEstimator: Step: 200, Epoch: 1, Steps/sec: 440.5841906691682
    FastEstimator: Step: 300, Epoch: 1, Steps/sec: 458.66033407201814
    FastEstimator: Step: 400, Epoch: 1, Steps/sec: 423.8592310935567
    FastEstimator: Step: 500, Epoch: 1, Steps/sec: 457.58897449238594
    FastEstimator: Step: 600, Epoch: 1, Steps/sec: 439.4676858001863
    FastEstimator: Step: 700, Epoch: 1, Steps/sec: 412.746418382437

