## Scale
```python
Scale(scalar, inputs=None, outputs=None, mode=None)
```
Preprocessing class for scaling dataset

#### Args:

* **scalar** :  Scalar for scaling the data

### forward
```python
forward(self, data, state)
```
Scales the data tensor

#### Args:

* **data** :  Data to be scaled
* **state** :  A dictionary containing background information such as 'mode'

#### Returns:
            Scaled data array        