## Minmax
```python
Minmax(inputs=None, outputs=None, mode=None, epsilon=1e-07)
```
Normalize data using the minmax method    

### forward
```python
forward(self, data, state)
```
Normalizes the data

#### Args:

* **data** :  Data to be normalized
* **state** :  A dictionary containing background information such as 'mode'

#### Returns:
            Normalized numpy array        