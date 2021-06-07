## BatchDataset<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/batch_dataset.py/#L27-L240>View source on Github</a>
```python
BatchDataset(
	datasets: Union[fastestimator.dataset.dataset.FEDataset, Iterable[fastestimator.dataset.dataset.FEDataset]],
	num_samples: Union[int, Iterable[int]],
	probability: Union[Iterable[float], NoneType]=None
)
-> None
```
BatchDataset extracts a list (batch) of data from a single dataset or multiple datasets.

This dataset helps to enable several use-cases:
1. Creating an unpaired dataset from two or more completely disjoint (no common keys) datasets.
    ```python
    ds1 = fe.dataset.DirDataset(...)  # {"a": <32x32>}
    ds2 = fe.dataset.DirDataset(...)  # {"b": <28x28>}
    unpaired_ds = fe.dataset.BatchDataset(datasets=[ds1, ds2], num_samples=[4, 4])
    # {"a": <4x32x32>, "b": <4x28x28>}
    ```
2. Deterministic class balanced sampling from two or more similar (all keys in common) datasets.
    ```python
    class1_ds = fe.dataset.DirDataset(...)  # {"x": <32x32>, "y": <>}
    class2_ds = fe.dataset.DirDataset(...)  # {"x": <32x32>, "y": <>}
    ds = fe.dataset.BatchDataset(datasets=[ds1, ds2], num_samples=[3, 5])
    # {"x": <8x32x32>, "y": <8>}  (3 of the samples are from class1_ds, 5 of the samples from class2_ds)
    ```
3. Probabilistic class balanced sampling from two or more similar (all keys in common) datasets.
    ```python
    class1_ds = fe.dataset.DirDataset(...)  # {"x": <32x32>, "y": <>}
    class2_ds = fe.dataset.DirDataset(...)  # {"x": <32x32>, "y": <>}
    ds = fe.dataset.BatchDataset(datasets=[ds1, ds2], num_samples=8, probability=[0.7, 0.3])
    # {"x": <8x32x32>, "y": <8>}  (~70% of the samples are from class1_ds, ~30% of the samples from class2_ds)
    ```


<h3>Args:</h3>


* **datasets**: The dataset(s) to use for batch sampling. While these should be FEDatasets, pytorch datasets will technically also work. If you use them, however, you will lose the .split() and .summary() methods.

* **num_samples**: Number of samples to draw from the `datasets`. May be a single int if used in conjunction with `probability`, otherwise a list of ints of len(`datasets`) is required.

* **probability**: Probability to draw from each dataset. Only allowed if `num_samples` is an integer.

---

### reset_index_maps<span class="tag">method of BatchDataset</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/batch_dataset.py/#L226-L240>View source on Github</a>
```python
reset_index_maps(
	self
)
-> None
```
Rearrange the index maps of this BatchDataset.

This method is invoked every epoch by OpDataset which allows each epoch to have different random pairings of the
basis datasets.

---

### split<span class="tag">method of BatchDataset</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/batch_dataset.py/#L118-L167>View source on Github</a>
```python
split(
	self, *fractions: Union[float, int, Iterable[int]])
-> Union[_ForwardRef('BatchDataset'), List[_ForwardRef('BatchDataset'
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


* ***fractions**: Floating point values will be interpreted as percentages, integers as an absolute number of datapoints, and an iterable of integers as the exact indices of the data that should be removed in order to create the new dataset. 

<h4>Raises:</h4>


* **NotImplementedError**: If the user created this dataset using one or more non-FEDataset inputs.

<h4>Returns:</h4>

<ul class="return-block"><li>    One or more new datasets which are created by removing elements from the current dataset. The number of
    datasets returned will be equal to the number of <code>fractions</code> provided. If only a single value is provided
    then the return will be a single dataset rather than a list of datasets.

</li></ul>

---

### summary<span class="tag">method of BatchDataset</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/batch_dataset.py/#L172-L182>View source on Github</a>
```python
summary(
	self
)
-> fastestimator.dataset.dataset.DatasetSummary
```
Generate a summary representation of this dataset.

<h4>Returns:</h4>

<ul class="return-block"><li>    A summary representation of this dataset.</li></ul>

