# tf.dataset.data.shakespeare<span class="tag">module</span>

---

## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/dataset/data/shakespeare.py/#L28-L68>View source on Github</a>
```python
load_data(
	root_dir: Union[str, NoneType]=None,
	seq_length: int=100
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, List[str]]
```
Load and return the Shakespeare dataset.

Shakespeare dataset is a collection of texts written by Shakespeare.
Sourced from https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt


<h3>Args:</h3>


* **root_dir**: The path to store the downloaded data. When `path` is not provided, the data will be saved into `fastestimator_data` under the user's home directory.

* **seq_length**: Length of data sequence. 

<h3>Returns:</h3>

<ul class="return-block"><li>    (train_data, vocab)</li></ul>

