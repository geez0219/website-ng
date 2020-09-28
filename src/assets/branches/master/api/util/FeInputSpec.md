## FeInputSpec
```python
FeInputSpec(
	model_input: Any,
	model: ~Model
)
```
A class to keep track of a model's input so that fake inputs can be generated.

This class is intentionally not @traceable.


#### Args:

* **model_input** :  The input to the model.
* **model** :  The model which corresponds to the given `model_input`.

### get_dummy_input
```python
get_dummy_input(
	self
)
-> Any
```
Get fake input for the model.


#### Returns:
    Input of the correct shape and dtype for the model.