## InMemoryDataset
```python
InMemoryDataset(*args, **kwargs)
```
A dataset abstraction to simplify the implementation of datasets which hold their data in memory.


#### Args:

* **data** :  A dictionary like {data_index {<instance dictionary>}}.

### summary
```python
summary(self) -> dataset.dataset.DatasetSummary
```
Generate a summary representation of this dataset.

#### Returns:
    A summary representation of this dataset.