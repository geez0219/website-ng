## Estimator<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/estimator.py/#L46-L442>View source on Github</a>
```python
Estimator(
	pipeline: fastestimator.pipeline.Pipeline,
	network: fastestimator.network.BaseNetwork,
	epochs: int,
	max_train_steps_per_epoch: Union[int, NoneType]=None,
	max_eval_steps_per_epoch: Union[int, NoneType]=None,
	traces: Union[NoneType, fastestimator.trace.trace.Trace, fastestimator.schedule.schedule.Scheduler[fastestimator.trace.trace.Trace], Iterable[Union[fastestimator.trace.trace.Trace, fastestimator.schedule.schedule.Scheduler[fastestimator.trace.trace.Trace]]]]=None,
	log_steps: Union[int, NoneType]=100,
	monitor_names: Union[NoneType, str, Iterable[str]]=None
)
```
One class to rule them all.

Estimator is the highest level class within FastEstimator. It is the class which is invoked to actually train
(estimator.fit) or test (estimator.test) models. It wraps `Pipeline`, `Network`, `Trace` objects together and
defines the whole optimization process.


<h3>Args:</h3>


* **pipeline**: An fe.Pipeline object that defines the data processing workflow.

* **network**: An fe.Network object that contains models and other training graph definitions.

* **epochs**: The number of epochs to run.

* **max_train_steps_per_epoch**: Training will complete after n steps even if loader is not yet exhausted. If None, all data will be used.

* **max_eval_steps_per_epoch**: Evaluation will complete after n steps even if loader is not yet exhausted. If None, all data will be used.

* **traces**: What Traces to run during training. If None, only the system's default Traces will be included.

* **log_steps**: Frequency (in steps) for printing log messages. 0 to disable all step-based printing (though epoch information will still print). None to completely disable printing.

* **monitor_names**: Additional keys from the data dictionary to be written into the logs.

---

### fit<span class="tag">method of Estimator</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/estimator.py/#L104-L124>View source on Github</a>
```python
fit(
	self,
	summary: Union[str, NoneType]=None,
	warmup: Union[bool, str]=True
)
-> Union[fastestimator.summary.summary.Summary, NoneType]
```
Train the network for the number of epochs specified by the estimator's constructor.


<h4>Args:</h4>


* **summary**: A name for the experiment. If provided, the log history will be recorded in-memory and returned as a summary object at the end of training.

* **warmup**: Whether to perform warmup before training begins. The warmup procedure will test one step at every epoch where schedulers cause the execution graph to change. This can take some time up front, but can also save significant heartache on epoch 300 when the training unexpectedly fails due to a tensor size mismatch. When set to "debug", the warmup will be performed in eager execution for easier debugging. 

<h4>Returns:</h4>

<ul class="return-block"><li>    A summary object containing the training history for this session iff a <code>summary</code> name was provided.</li></ul>

---

### get_scheduled_items<span class="tag">method of Estimator</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/estimator.py/#L224-L233>View source on Github</a>
```python
get_scheduled_items(
	self,
	mode: str
)
-> List[Any]
```
Get a list of items considered for scheduling.


<h4>Args:</h4>


* **mode**: Current execution mode. 

<h4>Returns:</h4>

<ul class="return-block"><li>    List of schedulable items in estimator.</li></ul>

---

### test<span class="tag">method of Estimator</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/estimator.py/#L158-L173>View source on Github</a>
```python
test(
	self,
	summary: Union[str, NoneType]=None
)
-> Union[fastestimator.summary.summary.Summary, NoneType]
```
Run the pipeline / network in test mode for one epoch.


<h4>Args:</h4>


* **summary**: A name for the experiment. If provided, the log history will be recorded in-memory and returned as a summary object at the end of training. If None, the default value will be whatever `summary` name was most recently provided to this Estimator's .fit() or .test() methods. 

<h4>Returns:</h4>

<ul class="return-block"><li>    A summary object containing the training history for this session iff the <code>summary</code> name is not None (after
    considering the default behavior above).</li></ul>

