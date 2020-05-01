# Tutorial 2: Creating a FastEstimator dataset

## Overview
Welcome to FastEstimator! In this tutorial we are going to cover three different ways to create a Dataset using FastEstimator. To showcase that, this tutorial is structured as follows:
* Torch Dataset Recap
* FastEstimator Dataset
	* Dataset from disk
		* LabeledDirDataset
		* CSVDataset
	* Dataset from memory
		* NumpyDataset
	* Dataset from generator

##  Torch Dataset Recap

A Dataset in FastEstimator is a class that wraps raw input data and makes it easier to ingest by your code. In this tutorial we will learn about the different ways we can create these Datasets.

The FastEstimator Dataset class inherited from the PyTorch Dataset class which provides a clean and efficient interface to load raw data. Thus, all your code that worked for PyTorch will continue to work for FastEstimator too. For a refresher on the PyTorch Dataset you can go [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

In this tutorial we will focus on two key functionalities that we need to provide for the Dataset class. The first one is the ability to get an individual data entry from the Dataset and the second one is the ability to get the length of the Dataset. This is done as follows:
* len(dataset) should return the size of the dataset.
* dataset[i] should return the ith sample in the dataset.

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
print (ds[0])
print (len(ds))
```

    {'x': array([0.04542747, 0.4162909 , 0.24907324, 0.84718429, 0.03261794]), 'y': 0.7262192218518535}
    100


## FastEstimator Dataset

In this section we will showcase how a Dataset can be created using FastEstimator. This tutorial shows three ways you can create a Dataset in FastEstimator, first is using data from disk, the second one showing the creation of Dataset from memory and the last one uses a generator to create the Dataset.

### 1. Dataset from disk

In this tutorial we will showcase two ways to create a Dataset from disk:

#### 1.1 LabeledDirDataset

To showcase this we will first have to create a dummy directory structure representing the two classes. Then we create a few files in each of the directories. The following image shows how the temp directory structure looks like:

<img src="assets/tutorial/../resources/t02_dataset_folder_structure.png" alt="drawing" width="200"/>

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

Once that is done, all you have to do is create a Dataset by passing the dummy directory to the LabeledDirDataset class constructor. The following code snippet shows how this can be done:


```python
dataset = fe.dataset.LabeledDirDataset(root_dir=tmpdirname)

print (dataset[0])
print (len(dataset))
```

    {'x': '/tmp/tmpa70w1cfg/tmpg___5e05/a1.txt', 'y': 0}
    4


#### 1.2 CSVDataset

To showcase creating Dataset from CSV we now create a dummy CSV file representing information for the two classes. First, let's create the data to be used as input as follows:


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

Once that is done you can create a Dataset by passing the CSV to the CSVDataset class constructor. The following code snippet shows how this can be done:


```python
dataset = fe.dataset.CSVDataset(file_path=os.path.join(tmpdirname, 'data.csv'))

print (dataset[0])
print (len(dataset))
```

    {'x': 'a1.txt', 'y': 0}
    4


### 2. Dataset from memory

It is also possible to create a Dataset from data stored in memory. The following two sections demonstrate how.

#### 2.1 NumpyDataset

To create a Dataset from memory, you use the NumpyDataset class passing it the data dictionary. The following code snippet shows how this can be done:


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


### 3. Dataset from Generator

It is also possible to create a Dataset using generators. As an example, we will first create a generator which will generate random input data for us.


```python
import numpy as np
def inputs():
    while True:
        yield {'x': np.random.rand(4), 'y':np.random.randint(2)}
```

We then pass the generator as an argument to the GeneratorDataset class. The following code snippet showcases how this can be done:


```python
from fastestimator.dataset import GeneratorDataset

dataset = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
print (dataset[0])
print (len(dataset))
```

    {'x': array([0.59146293, 0.53450084, 0.70742744, 0.05188558]), 'y': 0}
    10

