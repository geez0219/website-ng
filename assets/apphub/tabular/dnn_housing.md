# Boston housing price predictor (regression) using DNN

## Step 1: Prepare training and evaluation dataset, create FastEstimator Pipeline

Pipeline can take both data in memory and data in disk. In this example, we are going to use data in memory by loading data with tf.keras.datasets.boston_housing.
The following can be used to get the description of the data:


```python
from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()
print(boston.DESCR)
print(pd.DataFrame(boston.data).head())
```

    .. _boston_dataset:
    
    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    .. topic:: References
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
    
             0     1     2    3      4      5     6       7    8      9    10  \
    0  0.00632  18.0  2.31  0.0  0.538  6.575  65.2  4.0900  1.0  296.0  15.3   
    1  0.02731   0.0  7.07  0.0  0.469  6.421  78.9  4.9671  2.0  242.0  17.8   
    2  0.02729   0.0  7.07  0.0  0.469  7.185  61.1  4.9671  2.0  242.0  17.8   
    3  0.03237   0.0  2.18  0.0  0.458  6.998  45.8  6.0622  3.0  222.0  18.7   
    4  0.06905   0.0  2.18  0.0  0.458  7.147  54.2  6.0622  3.0  222.0  18.7   
    
           11    12  
    0  396.90  4.98  
    1  396.90  9.14  
    2  392.83  4.03  
    3  394.63  2.94  
    4  396.90  5.33  


The same data is also availabe from tensorflow without description and with train and eval data already separated


```python
import tensorflow as tf

(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.boston_housing.load_data()
print("train shape is {}".format(x_train.shape))
print("train value shape is {}".format(y_train.shape))
print("eval shape is {}".format(x_eval.shape))
print("eval value shape is {}".format(y_eval.shape))
```

    train shape is (404, 13)
    train value shape is (404,)
    eval shape is (102, 13)
    eval value shape is (102,)


Next we need to scale the inputs to the neural network. This is done by using a StandardScaler.


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_eval = scaler.transform(x_eval)
```

For in-memory data in Pipeline, the data format should be a nested dictionary like: {"mode1": {"feature1": numpy_array, "feature2": numpy_array, ...}, ...}. Each mode can be either train or eval, in our case, we have both train and eval. feature is the feature name, in our case, we have x and y. The network prediction will be a rank-1 array, in order to match prediction, we will expand the groud truth dimension by 1.


```python
import numpy as np

train_data = {"x": x_train, "y": np.expand_dims(y_train, -1)}
eval_data = {"x": x_eval, "y": np.expand_dims(y_eval, -1)}
data = {"train": train_data, "eval": eval_data}
```


```python
#Parameters
epochs = 50
batch_size = 32
steps_per_epoch = None
validation_steps = None
```

Now we are ready to define Pipeline:


```python
import fastestimator as fe

pipeline = fe.Pipeline(batch_size=batch_size, data=data)
```

## Step 2: Prepare model, create FastEstimator Network

First, we have to define the network architecture in tf.keras.Model or tf.keras.Sequential. After defining the architecture, users are expected to feed the architecture definition and its associated model name, optimizer and loss name (default to be 'loss') to FEModel.


```python
from tensorflow.keras import layers

def create_dnn():
    model = tf.keras.Sequential()

    model.add(layers.Dense(64, activation="relu", input_shape=(13,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="linear"))

    return model

model = fe.build(model_def=create_dnn, model_name="dnn", optimizer="adam", loss_name="loss")
```

Now we are ready to define the Network: given with a batch data with key x and y, we have to work our way to loss with series of operators. ModelOp is an operator that contains a model.


```python
from fastestimator.op.tensorop import ModelOp, MeanSquaredError

network = fe.Network(
    ops=[ModelOp(inputs="x", model=model, outputs="y_pred"), MeanSquaredError(inputs=("y","y_pred"),outputs="loss")])
```

## Step 3: Configure training, create Estimator

During the training loop, we want to: 1) measure lowest loss for data data 2) save the model with lowest valdiation loss. Trace class is used for anything related to training loop, we will need to import the ModelSaver trace.


```python
import tempfile
from fastestimator.trace import ModelSaver

model_dir = tempfile.mkdtemp()
traces = [ModelSaver(model_name="dnn", save_dir=model_dir, save_best=True)]
```

Now we can define the Estimator and specify the training configuation:


```python
estimator = fe.Estimator(network=network, 
                         pipeline=pipeline, 
                         epochs=epochs, 
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps,
                         traces=traces)
```

## Start Training


```python
estimator.fit()
```

## Inferencing

After training, the model is saved to a temporary folder. we can load the model from file and do inferencing on a sample.


```python
import os

model_path = os.path.join(model_dir, 'dnn_best_loss.h5')
trained_model = tf.keras.models.load_model(model_path, compile=False)
```

Randomly get one sample from validation set and compare the predicted value with model's prediction:


```python
import numpy as np

selected_idx = np.random.randint(0, high=101)
print("test sample idx {}, ground truth: {}".format(selected_idx, y_eval[selected_idx]))

test_sample = np.expand_dims(x_eval[selected_idx], axis=0)

predicted_value = trained_model.predict(test_sample)
print("model predicted value is {}".format(predicted_value))
```

    test sample idx 2, ground truth: 19.0
    model predicted value is [[20.183064]]

