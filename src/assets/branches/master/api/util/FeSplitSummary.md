## FeSplitSummary
```python
FeSplitSummary()
```
A class to summarize splits performed on an FE Dataset.

This class is intentionally not @traceable.

### add_split
```python
add_split(self, parent:Union[fastestimator.util.util.FEID, str], fraction:str) -> None
```
Record another split on this dataset.


#### Args:

* **parent** :  The id of the parent involved in the split (or 'self' if you are the parent).
* **fraction** :  The string representation of the split fraction that was used.

### dumps
```python
dumps(self) -> str
```
Generate a LaTeX formatted representation of this object.


#### Returns:
    A LaTeX string representation of this object.