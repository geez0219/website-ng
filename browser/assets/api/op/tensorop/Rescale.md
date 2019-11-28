## Rescale
```python
Rescale(inputs=None, outputs=None, mode=None)
```
Rescaling data according to

#### Args:

* **inputs** :  Name of the key in the dataset that is to be filtered.
* **outputs** :  Name of the key to be created/used in the dataset to store the results.
* **mode** :  mode that the filter acts on.    

### forward
```python
forward(self, data, state)
```
Scales the data tensor.

#### Args:

* **data** :  Data to be scaled.
* **state** :  Information about the current execution context.

#### Returns:
            Scaled data tensor        