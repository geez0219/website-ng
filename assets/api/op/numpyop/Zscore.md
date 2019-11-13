## Zscore
```python
Zscore(inputs=None, outputs=None, mode=None, epsilon=1e-07)
```
Standardize data using zscore method    

### forward
```python
forward(self, data, state)
```
Standardizes the data

#### Args:

* **data** :  Data to be standardized
* **state** :  A dictionary containing background information such as 'mode'

#### Returns:
            Array containing standardized data        