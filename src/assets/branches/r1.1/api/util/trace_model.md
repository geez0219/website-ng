## trace_model<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/util/traceability_util.py/#L984-L1014>View source on Github</a>
```python
trace_model(
	model: ~Model,
	model_idx: int,
	model_fn: Any,
	optimizer_fn: Any,
	weights_path: Any
)
-> ~Model
```
A function to add traceability information to an FE-compiled model.


<h3>Args:</h3>

* **model** :  The model to be made traceable.
* **model_idx** :  Which of the return values from the `model_fn` is this model (or -1 if only a single return value).
* **model_fn** :  The function used to generate this model.
* **optimizer_fn** :  The thing used to define this model's optimizer.
* **weights_path** :  The path to the weights for this model.

<h3>Returns:</h3>
    The `model`, but now with an fe_summary() method.

