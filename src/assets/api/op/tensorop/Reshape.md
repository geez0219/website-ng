## Reshape
```python
Reshape(shape, inputs=None, outputs=None, mode=None)
```
Preprocessing class for reshaping the data.

#### Args:

* **shape** :  target shape.
* **inputs** :  Name of the key in the dataset that is to be filtered.
* **outputs** :  Name of the key to be created/used in the dataset to store the results.
* **mode** :  mode that the filter acts on.

### forward
```python
forward(self, data, state)
```
Reshapes data tensor.

#### Args:

* **data** :  Data to be reshaped.
* **state** :  Information about the current execution context.

#### Returns:
            Reshaped tensor.        