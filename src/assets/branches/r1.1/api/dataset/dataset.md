# fastestimator.dataset.dataset<span class="tag">module</span>
---
## DatasetSummary<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/dataset.py/#L57-L94>View source on Github</a>
```python
DatasetSummary(
	num_instances: int,
	keys: Dict[str, fastestimator.dataset.dataset.KeySummary],
	num_classes: Union[int, NoneType]=None,
	class_key_mapping: Union[Dict[str, Any], NoneType]=None,
	class_key: Union[str, NoneType]=None
)
```
This class contains information summarizing a dataset object.

This class is intentionally not @traceable.


<h3>Args:</h3>

* **num_instances** :  The number of data instances within the dataset (influences the size of an epoch).
* **num_classes** :  How many different classes are present.
* **keys** :  What keys does the dataset provide, along with summary information about each key.
* **class_key** :  Which key corresponds to class information (if known).
* **class_key_mapping** :  A mapping of the original class string values to the values which are output to the pipeline.



## FEDataset<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/dataset.py/#L98-L279>View source on Github</a>
```python
FEDataset()
```


### split<span class="tag">method of FEDataset</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/dataset.py/#L167-L247>View source on Github</a>
```python
split(
	self, *fractions: Union[float, int, Iterable[int]])
-> Union[_ForwardRef('FEDataset'), List[_ForwardRef('FEDataset'
)]]
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


<h4>Args:</h4>

 *fractions :  Floating point values will be interpreted as percentages, integers as an absolute number of        datapoints, and an iterable of integers as the exact indices of the data that should be removed in order        to create the new dataset.

<h4>Returns:</h4>
    One or more new datasets which are created by removing elements from the current dataset. The number of    datasets returned will be equal to the number of `fractions` provided. If only a single value is provided    then the return will be a single dataset rather than a list of datasets.

### summary<span class="tag">method of FEDataset</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/dataset.py/#L271-L276>View source on Github</a>
```python
summary(
	self
)
-> fastestimator.dataset.dataset.DatasetSummary
```
Generate a summary representation of this dataset.

<h4>Returns:</h4>
    A summary representation of this dataset.



## InMemoryDataset<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/dataset.py/#L283-L427>View source on Github</a>
```python
InMemoryDataset(
	data: Dict[int, Dict[str, Any]]
)
-> None
```
A dataset abstraction to simplify the implementation of datasets which hold their data in memory.


<h3>Args:</h3>

* **data** :  A dictionary like {data_index {<instance dictionary>}}.

### summary<span class="tag">method of InMemoryDataset</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/dataset.py/#L393-L427>View source on Github</a>
```python
summary(
	self
)
-> fastestimator.dataset.dataset.DatasetSummary
```
Generate a summary representation of this dataset.

<h4>Returns:</h4>
    A summary representation of this dataset.



## KeySummary<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/dataset.py/#L30-L54>View source on Github</a>
```python
KeySummary(
	dtype: str,
	num_unique_values: Union[int, NoneType]=None,
	shape: List[Union[int, NoneType]]=()
)
-> None
```
A summary of the dataset attributes corresponding to a particular key.

This class is intentionally not @traceable.


<h3>Args:</h3>

* **num_unique_values** :  The number of unique values corresponding to a particular key (if known).
* **shape** :  The shape of the vectors corresponding to the key. None is used in a list to indicate that a dimension is        ragged.
* **dtype** :  The data type of instances corresponding to the given key.



