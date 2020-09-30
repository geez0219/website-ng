## WrapText
```python
WrapText(
	data: Union[str, int, float],
	threshold: int
)
```
A class to convert strings or numbers to wrappable latex representation.

This class will first convert the data to string, and then to a wrappable latex representation if its length is too
long. This fixes an issue which prevents the first element placed into a latex X column from wrapping correctly.


#### Args:

* **data** :  Input data to be converted.
* **threshold** :  When the length of `data` is greater than `threshold`, the resulting string will be made wrappable.

#### Raises:

* **AssertionError** :  If `data` is not a string, int, or float.

### dumps
```python
dumps(
	self
)
-> str
```
Get a string representation of this cell.


#### Returns:
    A string representation of itself.