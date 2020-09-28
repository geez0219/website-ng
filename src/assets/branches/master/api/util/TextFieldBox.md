## TextFieldBox
```python
TextFieldBox(
	name: str,
	height: str='2.5cm'
)
```
A class to wrap TextFields into padded boxes for use in nesting within tables.


#### Args:

* **name** :  The name to assign to this TextField. It should be unique within the document since changes to one box        will impact all boxes with the same name.
* **height** :  How tall should the TextField box be? Note that it will be wrapped by 10pt space on the top and bottom.