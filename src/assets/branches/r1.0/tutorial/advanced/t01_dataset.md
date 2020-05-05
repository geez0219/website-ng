# Advanced Tutorial 1: Dataset

## Overview
In this tutorial, we will talk about the following topics:
* [Dataset Summary](#ta01summary)
* [Dataset Splitting](#ta01splitting)
    * [Random Fraction Split](#ta01rfs)
    * [Random Count Split](#ta01rcs)
    * [Index Split](#ta01is)
* [Global Dataset Editing](#ta01gde)
* [BatchDataset](#ta01bd)
    * [Deterministic Batching](#ta01deterministic)
    * [Distribution Batching](#ta01distribution)
    * [Unpaired Dataset](#ta01ud)
* [Related Apphub Examples](#ta01rae)

Before going through the tutorial, it is recommended to check [beginner tutorial 02](./tutorials/beginner/t02_dataset) for basic understanding of `dataset` from PyTorch and FastEstimator. We will talk about more details about `fe.dataset` API in this tutorial.

<a id='ta01summary'></a>

## Dataset summary
As we have mentioned in previous tutorial, users can import our inherited dataset class for easy use in `Pipeline`. But how do we know what keys are available in the dataset?   Well, obviously one easy way is just call `dataset[0]` and check the keys. However, there's a more elegant way to check information of dataset: `dataset.summary()`.


```python
from fastestimator.dataset.data.mnist import load_data
train_data, eval_data = load_data()
```


```python
train_data.summary()
```




    <DatasetSummary {'num_instances': 60000, 'keys': {'x': <KeySummary {'shape': [28, 28], 'dtype': 'uint8'}>, 'y': <KeySummary {'num_unique_values': 10, 'shape': [], 'dtype': 'uint8'}>}}>



Or even more simply, by invoking the print function:


```python
print(train_data)
```

    {"keys": {"x": {"dtype": "uint8", "shape": [28, 28]}, "y": {"dtype": "uint8", "num_unique_values": 10, "shape": []}}, "num_instances": 60000}


<a id='ta01splitting'></a>

## Dataset Splitting

Dataset splitting is nothing new in machine learning. In FastEstimator, users can easily split their data in different ways. 

<a id='ta01rfs'></a>

### Random Fraction Split
Let's say we want to randomly split 50% of the evaluation data into test data. This is easily accomplished:


```python
test_data = eval_data.split(0.5)
```

Or if I want to split evaluation data into two test datasets with 20% of the evaluation data each:


```python
test_data1, test_data2 = eval_data.split(0.2, 0.2)
```

<a id='ta01rcs'></a>

### Random Count Split
Sometimes instead of fractions, we want an actual number of examples to split; for example, randomly splitting 100 samples from the evaluation dataset:


```python
test_data3 = eval_data.split(100)
```

And of course, we can generate multiple datasets by providing multiple inputs:


```python
test_data4, test_data5 = eval_data.split(100, 100)
```

<a id='ta01is'></a>

### Index Split
There are times when we need to split the dataset in a specific way. For that, you can provide a list of indexes. For example, if we want to split the 0th, 1st and 100th element of evaluation dataset into new test set:


```python
test_data6 = eval_data.split([0,1,100])
```

If you just want continuous index, here's an easy way to provide index:


```python
test_data7 = eval_data.split(range(100))
```

Needless to say, you can provide multiple inputs too:


```python
test_data7, test_data8 = eval_data.split([0, 1 ,2], [3, 4, 5])
```

<a id='ta01gde'></a>

## Global Dataset Editting
In deep learning, we usually process the dataset batch by batch. However, when we are handling tabular data, we might need to apply some transformation globally before the training. For example, we may want to standardize the tabular data using `sklearn`:


```python
from fastestimator.dataset.data.breast_cancer import load_data
from sklearn.preprocessing import StandardScaler

train_data, eval_data = load_data()
scaler = StandardScaler()

train_data["x"] = scaler.fit_transform(train_data["x"])
eval_data["x"] = scaler.transform(eval_data["x"])
```

<a id='ta01bd'></a>

## BatchDataset

There might be scenarios where we need to combine multiple datasets together into one dataset in a specific way. Let's consider three such use-cases now:

<a id='ta01deterministic'></a>

### Deterministic Batching
Let's say we have `mnist` and `cifar` datasets, and want to combine them with a total batch size of 8. If we always want 4 examples from `mnist` and the rest from `cifar`:


```python
from fastestimator.dataset.data import mnist, cifar10
from fastestimator.dataset import BatchDataset

mnist_data, _ = mnist.load_data(image_key="x", label_key="y")
cifar_data, _ = cifar10.load_data(image_key="x", label_key="y")

dataset_deterministic = BatchDataset(datasets=[mnist_data, cifar_data], num_samples=[4,4])
# ready to use dataset_deterministic in Pipeline, you might need to resize them to have consistent shape
```

<a id='ta01distribution'></a>

### Distribution Batching
Some people prefer randomness in a batch. For example, given total batch size of 8, let's say we want 0.5 probability of `mnist` and the other 0.5 from `cifar`:


```python
from fastestimator.dataset.data import mnist, cifar10
from fastestimator.dataset import BatchDataset

mnist_data, _ = mnist.load_data(image_key="x", label_key="y")
cifar_data, _ = cifar10.load_data(image_key="x", label_key="y")

dataset_distribution = BatchDataset(datasets=[mnist_data, cifar_data], num_samples=8, probability=[0.5, 0.5])
# ready to use dataset_distribution in Pipeline, you might need to resize them to have consistent shape
```

<a id='ta01ud'></a>

### Unpaired Dataset
Some deep learning tasks require random unpaired datasets. For example, in image-to-image translation (like Cycle-GAN), the system needs to randomly sample one horse image and one zebra image for every batch. In FastEstimator, `BatchDataset` can also handle unpaired datasets. The only restriction is that: **keys from two different datasets must be unique for unpaired datasets**.

For example, let's sample one image from `mnist` and one image from `cifar` for every batch:


```python
from fastestimator.dataset.data import mnist, cifar10
from fastestimator.dataset import BatchDataset

mnist_data, _ = mnist.load_data(image_key="x_mnist", label_key="y_mnist")
cifar_data, _ = cifar10.load_data(image_key="x_cifar", label_key="y_cifar")

dataset_unpaired = BatchDataset(datasets=[mnist_data, cifar_data], num_samples=[1,1])
# ready to use dataset_unpaired in Pipeline
```

<a id='ta01rae'></a>

## Apphub Examples
You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [DNN](./examples/tabular/dnn)
