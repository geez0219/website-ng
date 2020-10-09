## Trace<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/trace.py/#L29-L130>View source on Github</a>
```python
Trace(
	inputs: Union[NoneType, str, Iterable[str]]=None,
	outputs: Union[NoneType, str, Iterable[str]]=None,
	mode: Union[NoneType, str, Iterable[str]]=None
)
-> None
```
Trace controls the training loop. Users can use the `Trace` base class to customize their own functionality.

Traces are invoked by the fe.Estimator periodically as it runs. In addition to the current data dictionary, they are
also given a pointer to the current `System` instance which allows access to more information as well as giving the
ability to modify or even cancel training. The order of function invocations is as follows:

``` plot
        Training:                                       Testing:

    on_begin                                            on_begin
        |                                                   |
    on_epoch_begin (train)  <------<                    on_epoch_begin (test)  <------<
        |                          |                        |                         |
    on_batch_begin (train) <----<  |                    on_batch_begin (test) <----<  |
        |                       |  |                        |                      |  |
    on_batch_end (train) >-----^   |                    on_batch_end (test) >------^  |
        |                          ^                        |                         |
    on_epoch_end (train)           |                    on_epoch_end (test) >---------^
        |                          |                        |
    on_epoch_begin (eval)          |                    on_end
        |                          ^
    on_batch_begin (eval) <----<   |
        |                      |   |
    on_batch_end (eval) >-----^    |
        |                          |
    on_epoch_end (eval) >----------^
        |
    on_end
```


<h3>Args:</h3>


* **inputs**: A set of keys that this trace intends to read from the state dictionary as inputs.

* **outputs**: A set of keys that this trace intends to write into the system buffer.

* **mode**: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

---

### on_batch_begin<span class="tag">method of Trace</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/trace.py/#L100-L106>View source on Github</a>
```python
on_batch_begin(
	self,
	data: fastestimator.util.data.Data
)
-> None
```
Runs at the beginning of each batch.


<h4>Args:</h4>


* **data**: A dictionary through which traces can communicate with each other or write values for logging.

---

### on_batch_end<span class="tag">method of Trace</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/trace.py/#L108-L114>View source on Github</a>
```python
on_batch_end(
	self,
	data: fastestimator.util.data.Data
)
-> None
```
Runs at the end of each batch.


<h4>Args:</h4>


* **data**: The current batch and prediction data, as well as any information written by prior `Traces`.

---

### on_begin<span class="tag">method of Trace</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/trace.py/#L84-L90>View source on Github</a>
```python
on_begin(
	self,
	data: fastestimator.util.data.Data
)
-> None
```
Runs once at the beginning of training or testing.


<h4>Args:</h4>


* **data**: A dictionary through which traces can communicate with each other or write values for logging.

---

### on_end<span class="tag">method of Trace</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/trace.py/#L124-L130>View source on Github</a>
```python
on_end(
	self,
	data: fastestimator.util.data.Data
)
-> None
```
Runs once at the end training.


<h4>Args:</h4>


* **data**: A dictionary through which traces can communicate with each other or write values for logging.

---

### on_epoch_begin<span class="tag">method of Trace</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/trace.py/#L92-L98>View source on Github</a>
```python
on_epoch_begin(
	self,
	data: fastestimator.util.data.Data
)
-> None
```
Runs at the beginning of each epoch.


<h4>Args:</h4>


* **data**: A dictionary through which traces can communicate with each other or write values for logging.

---

### on_epoch_end<span class="tag">method of Trace</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/trace.py/#L116-L122>View source on Github</a>
```python
on_epoch_end(
	self,
	data: fastestimator.util.data.Data
)
-> None
```
Runs at the end of each epoch.


<h4>Args:</h4>


* **data**: A dictionary through which traces can communicate with each other or write values for logging.

