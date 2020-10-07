## SiameseDirDataset<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/siamese_dir_dataset.py/#L26-L177>View source on Github</a>
```python
SiameseDirDataset(
	root_dir: str,
	data_key_left: str='x_a',
	data_key_right: str='x_b',
	label_key: str='y',
	percent_matching_data: float=0.5,
	label_mapping: Union[Dict[str, Any], NoneType]=None,
	file_extension: Union[str, NoneType]=None
)
```
A dataset which returns pairs of data.

This dataset reads files from a folder hierarchy like root/class(/es)/data.file. Data is returned in pairs,
where the label value is 1 if the data are drawn from the same class, and 0 otherwise. One epoch is defined as
the time it takes to visit every data point exactly once as the 'data_key_left'. Each data point may occur zero
or many times as 'data_key_right' within the same epoch. SiameseDirDataset.split() will split by class index
rather than by data instance index.


<h3>Args:</h3>

* **root_dir** :  The path to the directory containing data sorted by folders.
* **data_key_left** :  What key to assign to the first data element in the pair.
* **data_key_right** :  What key to assign to the second data element in the pair.
* **label_key** :  What key to assign to the label values in the data dictionary.
* **percent_matching_data** :  What percentage of the time should data be paired by class (label value = 1).
* **label_mapping** :  A dictionary defining the mapping to use. If not provided will map classes to int labels.
* **file_extension** :  If provided then only files ending with the file_extension will be included.

### one_shot_trial<span class="tag">method of SiameseDirDataset</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/siamese_dir_dataset.py/#L144-L167>View source on Github</a>
```python
one_shot_trial(
	self,
	n: int
)
-> Tuple[List[str], List[str]]
```
Generate one-shot trial data.

The similarity should be highest between the index 0 elements of the arrays.


<h4>Args:</h4>

* **n** :  The number of samples to draw for computing one shot accuracy. Should be <= the number of total classes.

<h4>Returns:</h4>
    ([class_a_instance_x, class_a_instance_x, class_a_instance_x, ...],    [class_a_instance_w, class_b_instance_y, class_c_instance_z, ...])

### summary<span class="tag">method of SiameseDirDataset</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/siamese_dir_dataset.py/#L169-L177>View source on Github</a>
```python
summary(
	self
)
-> fastestimator.dataset.dataset.DatasetSummary
```
Generate a summary representation of this dataset.

<h4>Returns:</h4>
    A summary representation of this dataset.



