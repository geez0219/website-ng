## IterJoin
```python
IterJoin(data:Any, token:str)
```
A class to convert an iterable to a latex representation.

The data of this class can be any type. Usually it is iterable due to its capability to setup interval string.


#### Args:

* **data** :  Data of the cell.
* **token** :  String to be added in the interval of data entries.

### dumps
```python
dumps(self) -> str
```
Get a string representation of this cell.


#### Returns:
    A string representation of itself.