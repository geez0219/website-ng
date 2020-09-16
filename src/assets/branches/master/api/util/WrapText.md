## WrapText
```python
WrapText(data:Union[str, int, float], threshold:int)
```
A class to convert strings or numbers to wrappable latex representation.

This class will first convert the data to string, and then to wrappable latex representation if its length is too
long. This is to fix the issue that first string or number is not wrappable in X column type.


#### Args:

* **data** :  Input data to be converted.
* **threshold** :  When the length of <data> is greater than <threshold>, the resulting string will be made wrappable

### dumps
```python
dumps(self) -> str
```
Get a string representation of this cell.


#### Returns:
    A string representation of itself.