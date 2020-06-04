## FEDataset
```python
FEDataset()
```


### split
```python
split(self, *fractions:Union[float, int, Iterable[int]]) -> Union[_ForwardRef('FEDataset'), List[_ForwardRef('FEDataset')]]
```
Split this dataset into multiple smaller datasets.

This function enables several types of splitting:
1. Splitting by fractions.
    ```python
    ds = fe.dataset.FEDataset(...)  # len(ds) == 1000
    ds2 = ds.split(0.1)  # len(ds) == 900, len(ds2) == 100
    ds3, ds4 = ds.split(0.1, 0.2)  # len(ds) == 630, len(ds3) == 90, len(ds4) == 180
    ```
2. Splitting by counts.
    ```python
    ds = fe.dataset.FEDataset(...)  # len(ds) == 1000
    ds2 = ds.split(100)  # len(ds) == 900, len(ds2) == 100
    ds3, ds4 = ds.split(90, 180)  # len(ds) == 630, len(ds3) == 90, len(ds4) == 180
    ```
3. Splitting by indices.
    ```python
    ds = fe.dataset.FEDataset(...)  # len(ds) == 1000
    ds2 = ds.split([87,2,3,100,121,158])  # len(ds) == 994, len(ds2) == 6
    ds3 = ds.split(range(100))  # len(ds) == 894, len(ds3) == 100
    ```


#### Args:

 *fractions :  Floating point values will be interpreted as percentages, integers as an absolute number of        datapoints, and an iterable of integers as the exact indices of the data that should be removed in order        to create the new dataset.

#### Returns:
    One or more new datasets which are created by removing elements from the current dataset. The number of    datasets returned will be equal to the number of `fractions` provided. If only a single value is provided    then the return will be a single dataset rather than a list of datasets.

### summary
```python
summary(self) -> dataset.dataset.DatasetSummary
```
Generate a summary representation of this dataset.

#### Returns:
    A summary representation of this dataset.