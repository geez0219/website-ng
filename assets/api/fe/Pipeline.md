## Pipeline
```python
Pipeline(data, batch_size, ops=None, read_feature=None, padded_batch=False, expand_dims=False, max_shuffle_buffer_mb=3000)
```
Class representing the data pipeline required for fastestimator

#### Args:

* **data** :  The input for the pipeline. This can be either a dictionary, a tfrecord path or a RecordWriter.
* **batch_size** :  Integer representing the batch size per device for training the model.
* **ops** :  List of fastestimator operations that needs to be applied on the data in the pipeline.
* **read_feature** :  List of features that should be used in training. If None all the features available are used.
* **padded_batch** :  Boolean representing if a batch should be padded or not.
* **expand_dims** :  Boolean representing if a batch dimensions should be expanded or not.
* **max_shuffle_buffer_mb** :  Maximum buffer size to shuffle data. This is used only if the number of examples are        more than that could fit in the buffer. Defaults to 3000.

### benchmark
```python
benchmark(self, mode='train', num_steps=1000, log_interval=100, current_epoch=0)
```
Runs benchmarks for the current epoch.

#### Args:

* **mode** :  can be either "train" or "eval".
* **num_steps** :  the number of steps to show the results for.
* **log_interval** : 
* **current_epoch** :  to specify the current epoch in the training.        

### get_global_batch_size
```python
get_global_batch_size(self, epoch)
```
Gets the global batch size for the current epoch. Batch size changes if there is a schedule which specifies a        change for the given epoch.

#### Args:

* **epoch** :  The epoch number in the training        

### prepare
```python
prepare(self)
```
Create the dataset used by the pipeline by running all the ops specified.        

### show_results
```python
show_results(self, mode='train', num_steps=1, current_epoch=0)
```
Processes the pipeline ops on the given input data.

#### Args:

* **mode** :  can be either "train" or "eval".
* **num_steps** :  the number of steps for the pipeline to run.
* **current_epoch** :  to specify the current epoch in the training. This is useful if you are using a schedule to                change the pipeline during training.        