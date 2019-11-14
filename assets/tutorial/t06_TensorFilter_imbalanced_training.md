# Tutorial 6:  Dealing with imbalanced dataset using TensorFilter

When your dataset is imbalanced, training data needs to maintain a certain distribution to make sure minority classes are not ommitted during the training. In FastEstimator, `TensorFilter` is designed for that purpose.

`TensorFilter` is a Tensor Operator that is used in `Pipeline` along with other tensor operators such as `MinMax` and `Resize`.

There are only two differences between `TensorFilter` and `TensorOp`: 
1. `TensorFilter` does not have outputs.
2. The forward function of `TensorFilter` produces a boolean value which indicates whether to keep the data or not.

## Step 0 - Data preparation *(same as tutorial 1)*


```python
# Import libraries
import numpy as np
import tensorflow as tf
import fastestimator as fe
from fastestimator.op.tensorop import Minmax

# Load data and create dictionaries
(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
train_data = {"x": np.expand_dims(x_train, -1), "y": y_train}
eval_data = {"x": np.expand_dims(x_eval, -1), "y": y_eval}
data = {"train": train_data, "eval": eval_data}
```

## Step 1 - Customize your own Filter...
In this example, we will get rid of all images that have a label smaller than 5.


```python
from fastestimator.op.tensorop import TensorFilter

# We create our filter in forward function, it's just our condition.
class MyFilter(TensorFilter):
    def forward(self, data, state):
        pass_filter = data >= 5
        return pass_filter

# We specify the filter in Pipeline ops list.
pipeline = fe.Pipeline(batch_size=32, data=data, ops=[MyFilter(inputs="y"), Minmax(inputs="x", outputs="x")])
```


```python
# Let's check our pipeline ops results with show_results
results = pipeline.show_results()
print("filtering out all data with label less than 5, the labels of current batch are:")
print(results[0]["y"])
```

    filtering out all data with label less than 5, the labels of current batch are:
    tf.Tensor([5 9 6 9 8 8 9 6 5 8 9 6 8 9 5 9 6 7 5 8 7 5 7 5 6 6 9 8 6 5 6 5], shape=(32,), dtype=uint8)


## ... or use a pre-built ScalarFilter

In FastEstimator, if user needs to filter out scalar values with a certain probability, one can use pre-built filter `ScalarFilter`.   
Let's filter out even numbers labels with 50% probability:


```python
from fastestimator.op.tensorop import ScalarFilter

# We specify the list of scalars to filter out and the probability to keep these scalars
pipeline = fe.Pipeline(batch_size=32, 
                       data=data, 
                       ops=[ScalarFilter(inputs="y", filter_value=[0, 2, 4, 6, 8], keep_prob=[0.5, 0.5, 0.5, 0.5, 0.5]), 
                            Minmax(inputs="x", outputs="x")])
```


```python
# Let's check our pipeline ops results with show_results
results = pipeline.show_results(num_steps=10)

for idx in range(10):
    batch_label = results[idx]["y"].numpy()
    even_count = 0
    odd_count = 0
    for elem in batch_label:
        if elem % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
    print("in batch number {}, there are {} odd labels and {} even labels".format(idx, odd_count, even_count))
```

    in batch number 0, there are 20 odd labels and 12 even labels
    in batch number 1, there are 21 odd labels and 11 even labels
    in batch number 2, there are 20 odd labels and 12 even labels
    in batch number 3, there are 22 odd labels and 10 even labels
    in batch number 4, there are 22 odd labels and 10 even labels
    in batch number 5, there are 22 odd labels and 10 even labels
    in batch number 6, there are 21 odd labels and 11 even labels
    in batch number 7, there are 20 odd labels and 12 even labels
    in batch number 8, there are 22 odd labels and 10 even labels
    in batch number 9, there are 25 odd labels and 7 even labels

