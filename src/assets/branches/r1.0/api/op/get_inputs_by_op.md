## get_inputs_by_op<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/op.py/#L75-L90>View source on Github</a>
```python
get_inputs_by_op(
	op: fastestimator.op.op.Op,
	store: Mapping[str, Any]
)
-> Any
```
Retrieve the necessary input data from the data dictionary in order to run an `op`.


<h3>Args:</h3>


* **op**: The op to run.

* **store**: The system's data dictionary to draw inputs out of. 

<h3>Returns:</h3>

<ul class="return-block"><li>    Input data to be fed to the <code>op</code> forward function.</li></ul>

