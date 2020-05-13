## Estimator
```python
Estimator(pipeline:fastestimator.pipeline.Pipeline, network:fastestimator.network.BaseNetwork, epochs:int, max_train_steps_per_epoch:Union[int, NoneType]=None, max_eval_steps_per_epoch:Union[int, NoneType]=None, traces:Union[NoneType, fastestimator.trace.trace.Trace, fastestimator.schedule.schedule.Scheduler[fastestimator.trace.trace.Trace], Iterable[Union[fastestimator.trace.trace.Trace, fastestimator.schedule.schedule.Scheduler[fastestimator.trace.trace.Trace]]]]=None, log_steps:Union[int, NoneType]=100, monitor_names:Union[NoneType, str, Iterable[str]]=None)
```
One class to rule them all.

Estimator is the highest level class within FastEstimator. It is the class which is invoked to actually train
(estimator.fit) or test (estimator.test) models. It wraps `Pipeline`, `Network`, `Trace` objects together and
defines the whole optimization process.


#### Args:

* **pipeline** :  An fe.Pipeline object that defines the data processing workflow.
* **network** :  An fe.Network object that contains models and other training graph definitions.
* **epochs** :  The number of epochs to run.
* **max_train_steps_per_epoch** :  Training will complete after n steps even if loader is not yet exhausted. If None,        all data will be used.
* **max_eval_steps_per_epoch** :  Evaluation will complete after n steps even if loader is not yet exhausted. If None,        all data will be used.
* **traces** :  What Traces to run during training. If None, only the system's default Traces will be included.
* **log_steps** :  Frequency (in steps) for printing log messages. 0 to disable all step-based printing (though epoch        information will still print). None to completely disable printing.
* **monitor_names** :  Additional keys from the data dictionary to be written into the logs.

### fit
```python
fit(self, summary:Union[str, NoneType]=None, warmup:bool=True) -> Union[fastestimator.summary.summary.Summary, NoneType]
```
Train the network for the number of epochs specified by the estimator's constructor.


#### Args:

* **summary** :  A name for the experiment. If provided, the log history will be recorded in-memory and returned as        a summary object at the end of training.
* **warmup** :  Whether to perform warmup before training begins. The warmup procedure will test one step at every        epoch where schedulers cause the execution graph to change. This can take some time up front, but can        also save significant heartache on epoch 300 when the training unexpectedly fails due to a tensor size        mismatch.

#### Returns:
    A summary object containing the training history for this session iff a `summary` name was provided.

### get_scheduled_items
```python
get_scheduled_items(self, mode:str) -> List[Any]
```
Get a list of items considered for scheduling.


#### Args:

* **mode** :  Current execution mode.

#### Returns:
    List of schedulable items in estimator.

### test
```python
test(self, summary:Union[str, NoneType]=None) -> Union[fastestimator.summary.summary.Summary, NoneType]
```
Run the pipeline / network in test mode for one epoch.


#### Args:

* **summary** :  A name for the experiment. If provided, the log history will be recorded in-memory and returned as        a summary object at the end of training. If None, the default value will be whatever `summary` name was        most recently provided to this Estimator's .fit() or .test() methods.

#### Returns:
    A summary object containing the training history for this session iff the `summary` name is not None (after    considering the default behavior above).