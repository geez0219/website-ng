## Reshape
```python
Reshape(shape, inputs=None, outputs=None, mode=None)
```
Preprocessing class for reshaping the data

#### Args:

* **shape** :  target shape

### forward
```python
forward(self, data, state)
```
Reshapes data array

#### Args:

* **data** :  Data to be reshaped
* **state** :  A dictionary containing background information such as 'mode'

#### Returns:
            Reshaped array        