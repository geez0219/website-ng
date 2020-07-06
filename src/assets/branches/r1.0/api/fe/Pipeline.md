## Pipeline
```python
Pipeline(train_data:Union[NoneType, ~DataSource, fastestimator.schedule.schedule.Scheduler[~DataSource]]=None, eval_data:Union[NoneType, ~DataSource, fastestimator.schedule.schedule.Scheduler[~DataSource]]=None, test_data:Union[NoneType, ~DataSource, fastestimator.schedule.schedule.Scheduler[~DataSource]]=None, batch_size:Union[NoneType, int, fastestimator.schedule.schedule.Scheduler[int]]=None, ops:Union[NoneType, fastestimator.op.numpyop.numpyop.NumpyOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.numpyop.numpyop.NumpyOp], List[Union[fastestimator.op.numpyop.numpyop.NumpyOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.numpyop.numpyop.NumpyOp]]]]=None, num_process:Union[int, NoneType]=None, drop_last:bool=False, pad_value:Union[int, float, NoneType]=None, collate_fn:Union[Callable, NoneType]=None)
```
A data pipeline class that takes care of data pre-processing.


#### Args:

* **train_data** :  The training data, or None if no training data is available.
* **eval_data** :  The evaluation data, or None if no evaluation data is available.
* **test_data** :  The testing data, or None if no evaluation data is available.
* **batch_size** :  The batch size to be used by the pipeline. NOTE This argument is only applicable when using a        FastEstimator Dataset.
* **ops** :  NumpyOps to be used for pre-processing. NOTE This argument is only applicable when using a FastEstimator        Dataset.
* **num_process** :  Number of CPU threads to use for data pre-processing. NOTE This argument is only applicable when        using a FastEstimator Dataset. None will default to the system CPU count. Multiprocessing can be disabled by        passing 0 here, which can be useful for debugging.
* **drop_last** :  Whether to drop the last batch if the last batch is incomplete.
* **pad_value** :  The padding value if batch padding is needed. None indicates that no padding is needed. NOTE This        argument is only applicable when using a FastEstimator Dataset.
* **collate_fn** :  Function to merge data into one batch with input being list of elements.

### benchmark
```python
benchmark(self, mode:str='train', epoch:int=1, num_steps:int=1000, log_interval:int=100) -> None
```
Benchmark the pipeline processing speed.


#### Args:

* **mode** :  The execution mode to benchmark. This can be 'train', 'eval' or 'test'.
* **epoch** :  The epoch index to benchmark. Note that epoch indices are 1-indexed.
* **num_steps** :  The maximum number of steps over which to perform the benchmark.
* **log_interval** :  The logging interval.

### get_epochs_with_data
```python
get_epochs_with_data(self, total_epochs:int, mode:str) -> Set[int]
```
Get a set of epoch indices that contains data given mode.


#### Args:

* **total_epochs** :  Total number of epochs.
* **mode** :  Current execution mode.

#### Returns:
    Set of epoch indices.

### get_loader
```python
get_loader(self, mode:str, epoch:int=1, shuffle:Union[bool, NoneType]=None) -> Union[torch.utils.data.dataloader.DataLoader, tensorflow.python.data.ops.dataset_ops.DatasetV2]
```
Get a data loader from the Pipeline for a given `mode` and `epoch`.


#### Args:

* **mode** :  The execution mode for the loader. This can be 'train', 'eval' or 'test'.
* **epoch** :  The epoch index for the loader. Note that epoch indices are 1-indexed.
* **shuffle** :  Whether to shuffle the data. If None, the value for shuffle is based on mode. NOTE This argument        is only used with FastEstimator Datasets.

#### Returns:
    A data loader for the given `mode` and `epoch`.

### get_modes
```python
get_modes(self, epoch:Union[int, NoneType]=None) -> Set[str]
```
Get the modes for which the Pipeline has data.


#### Args:

* **epoch** :  The current epoch index

#### Returns:
    The modes for which the Pipeline has data.

### get_results
```python
get_results(self, mode:str='train', epoch:int=1, num_steps:int=1, shuffle:bool=False) -> Union[List[Dict[str, Any]], Dict[str, Any]]
```
Get sample Pipeline outputs.


#### Args:

* **mode** :  The execution mode in which to run. This can be "train", "eval", or "test".
* **epoch** :  The epoch index to run. Note that epoch indices are 1-indexed.
* **num_steps** :  Number of steps (batches) to get.
* **shuffle** :  Whether to use shuffling.

#### Returns:
    A list of batches of Pipeline outputs.

### get_scheduled_items
```python
get_scheduled_items(self, mode:str) -> List[Any]
```
Get a list of items considered for scheduling.


#### Args:

* **mode** :  Current execution mode.

#### Returns:
    List of schedulable items in Pipeline.

### transform
```python
transform(self, data:Dict[str, Any], mode:str, epoch:int=1) -> Dict[str, Any]
```
Apply all pipeline operations on a given data instance for the specified `mode` and `epoch`.


#### Args:

* **data** :  Input data in dictionary format.
* **mode** :  The execution mode in which to run. This can be "train", "eval", "test" or "infer".
* **epoch** :  The epoch index to run. Note that epoch indices are 1-indexed.

#### Returns:
    The transformed data.