## Minmax
```python
Minmax(inputs=None, outputs=None, mode=None, epsilon=1e-07)
```
Normalize data using the minmax method.    

### forward
```python
forward(self, data, state)
```
Normalizes the data tensor.

#### Args:

* **data** :  Data to be normalized.
* **state** :  Information about the current execution context.

#### Returns:
            Tensor after minmax.        