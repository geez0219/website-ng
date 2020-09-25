# Tutorial 2: Creating a FastEstimator dataset

## Overview
In this tutorial we are going to cover three different ways to create a Dataset using FastEstimator. This tutorial is structured as follows:

* [Torch Dataset Recap](./tutorials/master/beginner/t02_dataset#t02Recap)
* [FastEstimator Dataset](./tutorials/master/beginner/t02_dataset#t02FEDS)
    * [Dataset from disk](./tutorials/master/beginner/t02_dataset#t02Disk)
        * [LabeledDirDataset](./tutorials/master/beginner/t02_dataset#t02LDirDs)
        * [CSVDataset](./tutorials/master/beginner/t02_dataset#t02CSVDS)
    * [Dataset from memory](./tutorials/master/beginner/t02_dataset#t02Memory)
        * [NumpyDataset](./tutorials/master/beginner/t02_dataset#t02Numpy)
    * [Dataset from generator](./tutorials/master/beginner/t02_dataset#t02Generator)
* [Related Apphub Examples](./tutorials/master/beginner/t02_dataset#t02Apphub)

<a id='t02Recap'></a>

##  Torch Dataset Recap

A Dataset in FastEstimator is a class that wraps raw input data and makes it easier to ingest into your model(s). In this tutorial we will learn about the different ways we can create these Datasets.

The FastEstimator Dataset class inherits from the PyTorch Dataset class which provides a clean and efficient interface to load raw data. Thus, any code that you have written for PyTorch will continue to work in FastEstimator too. For a refresher on PyTorch Datasets you can go [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

In this tutorial we will focus on two key functionalities that we need to provide for the Dataset class. The first one is the ability to get an individual data entry from the Dataset and the second one is the ability to get the length of the Dataset. This is done as follows:

* len(dataset) should return the size (number of samples) of the dataset.
* dataset[i] should return the i-th sample in the dataset. The return value should be a dictionary with data values keyed by strings.

Let's create a simple PyTorch Dataset which shows this functionality.


```python
import numpy as np
from torch.utils.data import Dataset

class mydataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    def __len__(self):
        return self.data['x'].shape[0]
    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}

a = {'x': np.random.rand(100,5), 'y': np.random.rand(100)}
ds = mydataset(a)
print(ds[0])
print(len(ds))
```

    {'x': array([0.11588935, 0.27958611, 0.45703942, 0.36171531, 0.66472315]), 'y': 0.5592775462425909}
    100


<a id='t02FEDS'></a>

## FastEstimator Dataset

In this section we will showcase how a Dataset can be created using FastEstimator. This tutorial shows three ways to create Datasets. The first uses data from disk, the second uses data already in memory, and the third uses a generator to create a Dataset.

<a id='t02Disk'></a>

### 1. Dataset from disk

In this tutorial we will showcase two ways to create a Dataset from disk:

<a id='t02LDirDs'></a>

#### 1.1 LabeledDirDataset

To showcase this we will first have to create a dummy directory structure representing the two classes. Then we create a few files in each of the directories. The following image shows the hierarchy of our temporary data directory:

<img src="assets/branches/master/tutorial/../resources/t02_dataset_folder_structure.png" alt="drawing" width="200"/>

Let's prepare the data according to the directory structure:


```python
import os
import tempfile

import fastestimator as fe

tmpdirname = tempfile.mkdtemp()

a_tmpdirname = tempfile.TemporaryDirectory(dir=tmpdirname)
b_tmpdirname = tempfile.TemporaryDirectory(dir=tmpdirname)

a1 = open(os.path.join(a_tmpdirname.name, "a1.txt"), "x")
a2 = open(os.path.join(a_tmpdirname.name, "a2.txt"), "x")

b1 = open(os.path.join(b_tmpdirname.name, "b1.txt"), "x")
b2 = open(os.path.join(b_tmpdirname.name, "b2.txt"), "x")
```

Once that is done, all you have to do is create a Dataset by passing the dummy directory to the `LabeledDirDataset` class constructor. The following code snippet shows how this can be done:


```python
dataset = fe.dataset.LabeledDirDataset(root_dir=tmpdirname)

print(dataset[0])
print(len(dataset))
```

    {'x': '/var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp8ttxesg9/tmpcmkfpq_n/a1.txt', 'y': 0}
    4


<a id='t02CSVDS'></a>

#### 1.2 CSVDataset

To showcase creating a Dataset based on a CSV file, we now create a dummy CSV file representing information for the two classes. First, let's create the data to be used as input as follows:


```python
import os
import tempfile
import pandas as pd

import fastestimator as fe

tmpdirname = tempfile.mkdtemp()

data = {'x': ['a1.txt', 'a2.txt', 'b1.txt', 'b2.txt'], 'y': [0, 0, 1, 1]}
df = pd.DataFrame(data=data)
df.to_csv(os.path.join(tmpdirname, 'data.csv'), index=False)
```

Once that is done you can create a Dataset by passing the CSV to the `CSVDataset` class constructor. The following code snippet shows how this can be done:


```python
dataset = fe.dataset.CSVDataset(file_path=os.path.join(tmpdirname, 'data.csv'))

print(dataset[0])
print(len(dataset))
```

    {'x': 'a1.txt', 'y': 0}
    4


<a id='t02Memory'></a>

### 2. Dataset from memory

It is also possible to create a Dataset from data stored in memory. This may be useful for smaller datasets.

<a id='t02Numpy'></a>

#### 2.1 NumpyDataset

If you already have data in memory in the form of a Numpy array, it is easy to convert this data into a FastEstimator Dataset. To accomplish this, simply pass your data dictionary into the `NumpyDataset` class constructor. The following code snippet demonstrates this:


```python
import numpy as np
import tensorflow as tf

import fastestimator as fe

(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
train_data = fe.dataset.NumpyDataset({"x": x_train, "y": y_train})
eval_data = fe.dataset.NumpyDataset({"x": x_eval, "y": y_eval})

print (train_data[0]['y'])
print (len(train_data))
```

    5
    60000


<a id='t02Generator'></a>

### 3. Dataset from Generator

It is also possible to create a Dataset using generators. As an example, we will first create a generator which will generate random input data for us.


```python
import numpy as np

def inputs():
    while True:
        yield {'x': np.random.rand(4), 'y':np.random.randint(2)}
```

We then pass the generator as an argument to the `GeneratorDataset` class:


```python
from fastestimator.dataset import GeneratorDataset

dataset = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
print(dataset[0])
print(len(dataset))
```

    {'x': array([0.30590938, 0.65189247, 0.37606477, 0.01100033]), 'y': 1}
    10


<a id='t02Apphub'></a>

## Apphub Examples
You can find some practical examples of the concepts described here in the following FastEstimator Apphubs:

* [UNET](./examples/master/semantic_segmentation/unet)
* [DCGAN](./examples/master/image_generation/dcgan)
* [Siamese Networks](./examples/master/one_shot_learning/siamese)
