## Scheduler
```python
Scheduler(*args, **kwds)
```
A class which can wrap things like Datasets and Ops to make their behavior epoch-dependent.    

### get_all_values
```python
get_all_values(self) -> List[Union[~T, NoneType]]
```
Get a list of all the possible values stored in the `Scheduler`.

#### Returns:
            A list of all the values stored in the `Scheduler`. This may contain None values.        

### get_current_value
```python
get_current_value(self, epoch:int) -> Union[~T, NoneType]
```
Fetch whichever of the `Scheduler`s elements is appropriate based on the current epoch.

#### Args:

* **epoch** :  The current epoch.

#### Returns:
            The element from the Scheduler to be used at the given `epoch`. This value might be None.        