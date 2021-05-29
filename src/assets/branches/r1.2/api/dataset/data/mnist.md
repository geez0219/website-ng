# tf.dataset.data.mnist<span class="tag">module</span>

---

## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/dataset/data/mnist.py/#L22-L35>View source on Github</a>
```python
load_data(
	image_key: str='x',
	label_key: str='y'
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset]
```
Load and return the MNIST dataset.


<h3>Args:</h3>


* **image_key**: The key for image.

* **label_key**: The key for label. 

<h3>Returns:</h3>

<ul class="return-block"><li>    (train_data, eval_data)</li></ul>

