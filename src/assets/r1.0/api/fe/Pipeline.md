## Pipeline
```python
Pipeline(train_data:Union[NoneType, ~DataSource, fastestimator.schedule.schedule.Scheduler[~DataSource]]=None, eval_data:Union[NoneType, ~DataSource, fastestimator.schedule.schedule.Scheduler[~DataSource]]=None, test_data:Union[NoneType, ~DataSource, fastestimator.schedule.schedule.Scheduler[~DataSource]]=None, batch_size:Union[NoneType, int, fastestimator.schedule.schedule.Scheduler[int]]=None, ops:Union[NoneType, fastestimator.op.numpyop.numpyop.NumpyOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.numpyop.numpyop.NumpyOp], List[Union[fastestimator.op.numpyop.numpyop.NumpyOp, fastestimator.schedule.schedule.Scheduler[fastestimator.op.numpyop.numpyop.NumpyOp]]]]=None, num_process:Union[int, NoneType]=None, drop_last:bool=False, pad_value:Union[int, float, NoneType]=None)
```
Data pipeline class that takes care of the data preprocessing.

#### Args:

* **train_data** :  training data, can be a tf.data.Dataset, fe.dataset or torch.data.DataLoader or a scheduler of them.                    Defaults to None, which means no training data available.
* **eval_data** :  evaludation data, can be a tf.data.Dataset, fe.dataset or torch.data.DataLoader or a scheduler of them.                    Defaults to None, which means no evaluation data available.
* **test_data** :  testing data, can be a tf.data.Dataset, fe.dataset or torch.data.DataLoader or a scheduler of them.                    Defaults to None, which means no testing data available.
* **batch_size** :  batch size, can be an integer or a scheduelr of integer, only used when fe.dataset is available.                    Defaults to None.
* **ops** :  preprocessing numpy ops, only used when fe.dataset is available. Defaults to None.
* **num_process** :  number of processes, only used whenfe.dataset is available. Defaults to None, which will be the                    system cpu count. use num_process=0 for debugging.
* **drop_last** :  whether to drop the last batch if last batch is incomplete.
* **pad_value** :  the padding value if batch padding is needed. Defaults to None, which indicates no padding. only used                    when fe.dataset is available.    

### benchmark
```python
benchmark(self, mode:str='train', epoch:int=1, num_steps:int=1000, log_interval:int=100)
```
benchmark the pipeline processing speed

#### Args:

* **mode** :  Current mode, can be 'train', 'eval' or 'test'.
* **epoch** :  Current epoch index. Defaults to 1.
* **num_steps** :  Maximum number of steps to do benchmark on. Defaults to 1000.
* **log_interval** :  Logging interval. Defaults to 100.        

### get_loader
```python
get_loader(self, mode:str, epoch:int=1, shuffle:Union[bool, NoneType]=None) -> Union[torch.utils.data.dataloader.DataLoader, tensorflow.python.data.ops.dataset_ops.DatasetV2]
```
get the data loader given mode and epoch

#### Args:

* **mode** :  Current mode, can be 'train', 'eval' or 'test'.
* **epoch** :  Current epoch index. Defaults to 1.
* **shuffle** :  Whether to shuffle, only used with FE dataset. If None, shuffle is based on mode. Defaults to None.

#### Returns:
            data loader given the mode and epoch.        

### get_modes
```python
get_modes(self) -> Set[str]
```
get the active modes in pipeline

#### Returns:
            set of active modes        

### get_results
```python
get_results(self, mode:str='train', epoch:int=1, num_steps:int=1, shuffle:bool=False) -> Union[List[Dict[str, Any]], Dict[str, Any]]
```
get the pipeline outputs after all ops

#### Args:

* **mode** :  Current mode, can be 'train', 'eval' or 'test'.
* **epoch** :  Current epoch index. Defaults to 1.
* **num_steps** :  number of steps(batches) to get. Defaults to 1.
* **shuffle** :  whether to use shuffling

#### Returns:
            pipeline outputs        

### get_signature_epochs
```python
get_signature_epochs(self, total_epochs:int)
```
get the signature epochs that scheduler will be effective on.

#### Args:

* **total_epochs** :  total number of epochs

#### Returns:

* **set** :  set of epoch index        

### transform
```python
transform(self, data:Dict[str, Any], mode:str, epoch:int=1) -> Dict[str, Any]
```
apply all pipeline operations on given data for certain mode and epoch.

#### Args:

* **data** :  Input data in dictionary format
* **mode** :  Current mode, can be "train", "eval", "test" or "infer"
* **epoch** :  Current epoch index. Defaults to 1.

#### Returns:
            transformed data        