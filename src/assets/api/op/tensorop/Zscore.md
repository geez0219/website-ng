## Zscore
```python
Zscore(inputs=None, outputs=None, mode=None, epsilon=1e-07)
```
Standardize data using zscore method.    

### forward
```python
forward(self, data, state)
```
Standardizes the data tensor.

#### Args:

* **data** :  Data to be standardized.
* **state** :  Information about the current execution context.

#### Returns:
            Tensor containing standardized data.        