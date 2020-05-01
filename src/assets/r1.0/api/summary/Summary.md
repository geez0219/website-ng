## Summary
```python
Summary(name:Union[str, NoneType]) -> None
```
A summary object that records training history.

#### Args:

* **name** :  Name of the experiment. If None then experiment results will be ignored    

### merge
```python
merge(self, other:'Summary')
```
Merge another `Summary` into this one.

#### Args:

* **other** :  Other `summary` object to be merged.        