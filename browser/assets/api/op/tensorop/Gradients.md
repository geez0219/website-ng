## Gradients
```python
Gradients(loss, models=None, keys=None, outputs=None)
```
This class computes gradients

#### Args:

* **loss (str)** :  The loss key to compute gradients from
* **models (keras.model, list)** :  A list of models to compute gradients against
* **keys (str, list)** :  A list of keys corresponding to variables to compute gradients against
* **outputs (str, list)** :  A list of output names (model gradients first, then key gradients)        