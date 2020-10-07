## PickleDataset<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/pickle_dataset.py/#L24-L37>View source on Github</a>
```python
PickleDataset(
	file_path: str
)
-> None
```
A dataset from a pickle file.

PickleDataset reads entries from pickled pandas data-frames. The root directory of the pickle file may be accessed
using dataset.parent_path. This may be useful if the file contains relative path information that you want to feed
into, say, an ImageReader Op.


<h3>Args:</h3>

* **file_path** :  The (absolute) path to the pickle file.



