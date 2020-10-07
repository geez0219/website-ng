## DirDataset<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/dir_dataset.py/#L23-L54>View source on Github</a>
```python
DirDataset(
	root_dir: str,
	data_key: str='x',
	file_extension: Union[str, NoneType]=None,
	recursive_search: bool=True
)
-> None
```
A dataset which reads files from a folder hierarchy like root/data.file.


<h3>Args:</h3>

* **root_dir** :  The path to the directory containing data.
* **data_key** :  What key to assign to the data values in the data dictionary.
* **file_extension** :  If provided then only files ending with the file_extension will be included.
* **recursive_search** :  Whether to search within subdirectories for files.



