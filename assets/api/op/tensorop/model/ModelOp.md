## ModelOp
```python
ModelOp(model, inputs=None, outputs=None, mode=None, track_input=False)
```
This class represents the Model operator that defines String keys for storing batch data and predictions

#### Args:

* **model** :  keras model compiled by fe.build
* **inputs** :  String key of input training data. Defaults to None.
* **outputs** :  String key of predictions. Defaults to None.
* **mode** :  'train' or 'eval'. Defaults to None.
* **track_input** :  If 'true' it tracks the gradients with respect to inputs. Defaults to False.