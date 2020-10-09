# tf.dataset.data.nih_chestxray<span class="tag">module</span>

---

## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/dataset/data/nih_chestxray.py/#L42-L86>View source on Github</a>
```python
load_data(
	root_dir: Union[str, NoneType]=None
)
-> fastestimator.dataset.dir_dataset.DirDataset
```
Load and return the NIH Chest X-ray dataset.


<h3>Args:</h3>


* **root_dir**: The path to store the downloaded data. When `path` is not provided, the data will be saved into `fastestimator_data` under the user's home directory. 

<h3>Returns:</h3>

<ul class="return-block"><li>    train_data</li></ul>

