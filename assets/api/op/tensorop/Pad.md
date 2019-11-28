## Pad
```python
Pad(padded_shape, inputs=None, outputs=None, mode=None)
```
Pad data to target padded shape

#### Args:

* **padded_shape** :  target padded shape
* **inputs** :  Name of the key in the dataset that is to be filtered.
* **outputs** :  Name of the key to be created/used in the dataset to store the results.
* **mode** :  mode that the filter acts on.    

### forward
```python
forward(self, data, state)
```
Pad data to target padded shape.

#### Args:

* **data** :  Data to be scaled.
* **state** :  Information about the current execution context.

#### Returns:
            padded data tensor        