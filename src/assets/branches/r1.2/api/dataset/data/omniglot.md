# tf.dataset.data.omniglot<span class="tag">module</span>

---

## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/dataset/data/omniglot.py/#L28-L63>View source on Github</a>
```python
load_data(
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.siamese_dir_dataset.SiameseDirDataset, fastestimator.dataset.siamese_dir_dataset.SiameseDirDataset]
```
Load and return the Omniglot dataset.


<h3>Args:</h3>


* **root_dir**: The path to store the downloaded data. When `path` is not provided, the data will be saved into `fastestimator_data` under the user's home directory. 

<h3>Returns:</h3>

<ul class="return-block"><li>    (train_data, eval_data)</li></ul>

