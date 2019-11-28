

### build
```python
build(model_def, model_name, optimizer, loss_name, custom_objects=None)
```
build keras model instance in FastEstimator

#### Args:

* **model_def (function)** :  function definition of tf.keras model or path of model file(h5)
* **model_name (str, list, tuple)** :  model name(s)
* **optimizer (str, optimizer, list, tuple)** :  optimizer(s)
* **loss_name (str, list, tuple)** :  loss name(s)
* **custom_objects (dict)** :  dictionary that maps custom

#### Returns:

* **model** :  model(s) compiled by FastEstimator