## get_inputs_by_op<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/op.py/#L77-L102>View source on Github</a>
```python
get_inputs_by_op(
	op: fastestimator.op.op.Op,
	store: Mapping[str, Any],
	copy_on_write: bool=False
)
-> Any
```
Retrieve the necessary input data from the data dictionary in order to run an `op`.


<h3>Args:</h3>


* **op**: The op to run.

* **store**: The system's data dictionary to draw inputs out of.

* **copy_on_write**: Whether to copy read-only data to make it writeable before returning it. 

<h3>Returns:</h3>

<ul class="return-block"><li>    Input data to be fed to the <code>op</code> forward function.</li></ul>

