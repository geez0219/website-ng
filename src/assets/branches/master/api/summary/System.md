## System
```python
System(network:fastestimator.network.BaseNetwork, pipeline:fastestimator.pipeline.Pipeline, traces:List[Union[_ForwardRef('Trace'), fastestimator.schedule.schedule.Scheduler[_ForwardRef('Trace')]]], mode:Union[str, NoneType]=None, num_devices:int=1, log_steps:Union[int, NoneType]=None, total_epochs:int=0, max_train_steps_per_epoch:Union[int, NoneType]=None, max_eval_steps_per_epoch:Union[int, NoneType]=None, system_config:Union[List[fastestimator.util.traceability_util.FeSummaryTable], NoneType]=None) -> None
```
A class which tracks state information while the fe.Estimator is running.

This class is intentionally not @traceable.


#### Args:

* **network** :  The network instance being used by the current fe.Estimator.
* **pipeline** :  The pipeline instance being used by the current fe.Estimator.
* **traces** :  The traces provided to the current fe.Estimator.
* **mode** :  The current execution mode (or None for warmup).
* **num_devices** :  How many GPUs are available for training.
* **log_steps** :  Log every n steps (0 to disable train logging, None to disable all logging).
* **total_epochs** :  How many epochs training is expected to run for.
* **max_train_steps_per_epoch** :  Whether training epochs will be cut short after N steps (or use None if they will run        to completion)
* **system_config** :  A description of the initialization parameters defining the associated estimator.
* **Attributes** : 
* **mode** :  What is the current execution mode of the estimator ('train', 'eval', 'test'), None if warmup.
* **global_step** :  How many training steps have elapsed.
* **num_devices** :  How many GPUs are available for training.
* **log_steps** :  Log every n steps (0 to disable train logging, None to disable all logging).
* **total_epochs** :  How many epochs training is expected to run for.
* **epoch_idx** :  The current epoch index for the training (starting from 1).
* **batch_idx** :  The current batch index within an epoch (starting from 1).
* **stop_training** :  A flag to signal that training should abort.
* **network** :  A reference to the network being used.
* **pipeline** :  A reference to the pipeline being used.
* **traces** :  The traces being used.
* **max_train_steps_per_epoch** :  Training will complete after n steps even if loader is not yet exhausted.
* **max_eval_steps_per_epoch** :  Evaluation will complete after n steps even if loader is not yet exhausted.
* **summary** :  An object to write experiment results to.
* **experiment_time** :  A timestamp indicating when this model was trained.

### load_state
```python
load_state(self, json_path) -> None
```
Load training state.


#### Args:

* **json_path** :  The json file path to load from.

### reset
```python
reset(self, summary_name:Union[str, NoneType]=None, system_config:Union[str, NoneType]=None) -> None
```
Reset the current `System` for a new round of training, including a new `Summary` object.


#### Args:

* **summary_name** :  The name of the experiment. The `Summary` object will store information iff name is not None.
* **system_config** :  A description of the initialization parameters defining the associated estimator.

### reset_for_test
```python
reset_for_test(self, summary_name:Union[str, NoneType]=None) -> None
```
Partially reset the current `System` object for a new round of testing.


#### Args:

* **summary_name** :  The name of the experiment. If not provided, the system will re-use the previous summary name.

### save_state
```python
save_state(self, json_path) -> None
```
Load training state.


#### Args:

* **json_path** :  The json file path to save to.

### update_batch_idx
```python
update_batch_idx(self) -> None
```
Increment the current `batch_idx`.
        

### update_global_step
```python
update_global_step(self) -> None
```
Increment the current `global_step`.
        

### write_summary
```python
write_summary(self, key:str, value:Any) -> None
```
Write an entry into the `Summary` object (iff the experiment was named).


#### Args:

* **key** :  The key to write into the summary object.
* **value** :  The value to write into the summary object.