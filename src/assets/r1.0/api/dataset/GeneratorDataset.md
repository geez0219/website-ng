## GeneratorDataset
```python
GeneratorDataset(generator:Generator[Dict[str, Any], int, NoneType], samples_per_epoch:int) -> None
```
A dataset from a generator function.

#### Args:

* **generator** :  The generator function to invoke in order to get a data sample.
* **samples_per_epoch** :  How many samples should be drawn from the generator during an epoch. Note that the generator            function will actually be invoke more times than the number specified here due to backend validation            routines.    

### summary
```python
summary(self) -> fastestimator.dataset.dataset.DatasetSummary
```
Generate a summary representation of this dataset.

#### Returns:
            A summary representation of this dataset.        