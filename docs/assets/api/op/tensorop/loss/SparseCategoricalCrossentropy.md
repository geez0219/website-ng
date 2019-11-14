## SparseCategoricalCrossentropy
```python
SparseCategoricalCrossentropy(y_true=None, y_pred=None, inputs=None, outputs=None, mode=None, **kwargs)
```
Calculate sparse categorical cross entropy, the rest of the keyword argument will be passed totf.losses.SparseCategoricalCrossentropy

#### Args:

* **y_true** :  ground truth label key
* **y_pred** :  prediction label key
* **inputs** :  A tuple or list like [<y_true>, <y_pred>]
* **outputs** :  Where to store the computed loss value (not required under normal use cases)
* **mode** :  'train', 'eval', 'test', or None
* **kwargs** :  Arguments to be passed along to the tf.losses constructor. Passing the 'reduction' arg will raise a            KeyError