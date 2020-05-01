# Advanced Tutorial 1: Dataset

## Overview
In this tutorial, we will talk about the following topics:
* Dataset summary
* Dataset splitting
    * Random fraction split
    * Random count split
    * Index split
* Global Dataset Editing
* BatchDataset
    * Deterministic batching
    * Distribution batching
    * Unpaired dataset

Before going through the tutorial, it is recommended to check [beginner tutorial 02](linkneeded) for basic understanding of `dataset` from Pytorch and FastEstimator. We will talk about more details about `fe.dataset` API in this tutorial.

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



## Dataset Splitting

dataset splitting is nothing new in machine learning, in FastEstimator, users can easily split their data in different ways:

### random fraction split
Let's say we want to randomly split 50% of evaluation data into test data, simply do:


```python
test_data = eval_data.split(0.5)
```

Or if I want to split evaluation data into two test datasets with 20% of evaluation data each.


```python
testdata1, test_data2 = eval_data.split(0.2, 0.2)
```

### Random count split
Sometimes instead of fractions, we want actual number of examples to split, for example, randomly splitting 100 samples from evaluation dataset:


```python
test_data3 = eval_data.split(100)
```

And of course, we can generate multiple datasets by providing multiple inputs:


```python
test_data4, test_data5 = eval_data.split(100, 100)
```

### Index split
There are times when we need to split the dataset in a specific way, then you can provide the index: for example, if we want to split the 0th, 1st and 100th element of evaluation dataset into new test set:


```python
test_data6 = eval_data.split([0,1,100])
```

if you just want continuous index, here's an easy way to provide index:


```python
test_data7 = eval_data.split(range(100))
```

Needless to say, you can provide multiple inputs too:


```python
test_data7, test_data8 = eval_data.split([0, 1 ,2], [3, 4, 5])
```

## Global Dataset Editting
In deep learning, we usually process the dataset batch by batch. However, when we are handling the tabular data, we might need to apply some transformation globally before the training.  For example, standardize the tabular data using `sklearn`:


```python
from fastestimator.dataset.data.breast_cancer import load_data
from sklearn.preprocessing import StandardScaler

train_data, eval_data = load_data()
scaler = StandardScaler()

train_data["x"] = scaler.fit_transform(train_data["x"])
eval_data["x"] = scaler.transform(eval_data["x"])
```

## BatchDataset

There might be scenarios where we need to combine multiple datasets together into one dataset in a specific way, next we will talk about 3 such use cases.

### Deterministic batching
Let's say we have `mnist` and `cifar` dataset, given the total batch size of 8, if we always want 4 examples from `mnist` and the rest from `cifar`:


```python
from fastestimator.dataset.data import mnist, cifar10
from fastestimator.dataset import BatchDataset

mnist_data, _ = mnist.load_data(image_key="x", label_key="y")
cifar_data, _ = cifar10.load_data(image_key="x", label_key="y")

dataset_deterministic = BatchDataset(datasets=[mnist_data, cifar_data], num_samples=[4,4])
# ready to use dataset_deterministic in Pipeline
```

### Distribution batching
Some people who prefer randomness in a batch, for example, given total batch size of 8, we want 0.5 probability of `mnist` and the other 0.5 from `cifar`:


```python
from fastestimator.dataset.data import mnist, cifar10
from fastestimator.dataset import BatchDataset

mnist_data, _ = mnist.load_data(image_key="x", label_key="y")
cifar_data, _ = cifar10.load_data(image_key="x", label_key="y")

dataset_distribution = BatchDataset(datasets=[mnist_data, cifar_data], num_samples=8, probability=[0.5, 0.5])
# ready to use dataset_distribution in Pipeline
```

### Unpaird dataset
Some deep learning tasks require random unpaired dataset. For example, in image-to-image translation (like Cycle-GAN), it needs to randomly sample one horse image and one zebra image for every batch. In FastEstimator, `BatchDataset` can also handle unpaired dataset, all you need to make sure is: **keys from two different datasets must be unique for unpaired dataset**.

For example, let's sample one image from `mnist` and one image from `cifar` for every batch:


```python
from fastestimator.dataset.data import mnist, cifar10
from fastestimator.dataset import BatchDataset

mnist_data, _ = mnist.load_data(image_key="x_mnist", label_key="y_mnist")
cifar_data, _ = cifar10.load_data(image_key="x_cifar", label_key="y_cifar")

dataset_unpaired = BatchDataset(datasets=[mnist_data, cifar_data], num_samples=[1,1])
# ready to use dataset_unpaired in Pipeline
```
