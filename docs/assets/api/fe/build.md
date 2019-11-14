

### build
```python
build(model_def, model_name, optimizer, loss_name)
```
build keras model instance in FastEstimator

#### Args:

* **model_def (function)** :  function definition of tf.keras model
* **model_name (str, list, tuple)** :  model name(s)
* **optimizer (str, optimizer, list, tuple)** :  optimizer(s)
* **loss_name (str, list, tuple)** :  loss name(s)

#### Returns:

* **model** :  model(s) compiled by FastEstimator