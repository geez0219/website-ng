

### trace_model
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


#### Args:

* **model** :  The model to be made traceable.
* **model_idx** :  Which of the return values from the `model_fn` is this model (or -1 if only a single return value).
* **model_fn** :  The function used to generate this model.
* **optimizer_fn** :  The thing used to define this model's optimizer.
* **weights_path** :  The path to the weights for this model.

#### Returns:
    The `model`, but now with an fe_summary() method.