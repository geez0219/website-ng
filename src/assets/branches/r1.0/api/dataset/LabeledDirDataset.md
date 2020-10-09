## LabeledDirDataset<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/dataset/labeled_dir_dataset.py/#L22-L81>View source on Github</a>
```python
LabeledDirDataset(
	root_dir: str,
	data_key: str='x',
	label_key: str='y',
	label_mapping: Union[Dict[str, Any], NoneType]=None,
	file_extension: Union[str, NoneType]=None
)
-> None
```
A dataset which reads files from a folder hierarchy like root/class(/es)/data.file.


<h3>Args:</h3>


* **root_dir**: The path to the directory containing data sorted by folders.

* **data_key**: What key to assign to the data values in the data dictionary.

* **label_key**: What key to assign to the label values in the data dictionary.

* **label_mapping**: A dictionary defining the mapping to use. If not provided will map classes to int labels.

* **file_extension**: If provided then only files ending with the file_extension will be included.

---

### summary<span class="tag">method of LabeledDirDataset</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/dataset/labeled_dir_dataset.py/#L72-L81>View source on Github</a>
```python
summary(
	self
)
-> fastestimator.dataset.dataset.DatasetSummary
```
Generate a summary representation of this dataset.

<h4>Returns:</h4>

<ul class="return-block"><li>    A summary representation of this dataset.</li></ul>

