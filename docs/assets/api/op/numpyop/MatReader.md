## MatReader
```python
MatReader(inputs=None, outputs=None, mode=None, parent_path='')
```
Class for reading .mat files.

#### Args:

* **parent_path** :  Parent path that will be added on given path.

### forward
```python
forward(self, data, state)
```
Reads mat file as dict.

#### Args:

* **data** :  Path to the mat file.
* **state** :  A dictionary containing background information such as 'mode'

#### Returns:
           dict        