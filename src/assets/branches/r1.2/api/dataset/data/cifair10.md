# tf.dataset.data.cifair10<span class="tag">module</span>

---

## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/dataset/data/cifair10.py/#L25-L72>View source on Github</a>
```python
load_data(
	image_key: str='x',
	label_key: str='y'
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset]
```
Load and return the ciFAIR10 dataset.

This is the cifar10 dataset but with test set duplicates removed and replaced. See
https://arxiv.org/pdf/1902.00423.pdf or https://cvjena.github.io/cifair/ for details. Cite the paper if you use the
dataset.


<h3>Args:</h3>


* **image_key**: The key for image.

* **label_key**: The key for label. 

<h3>Returns:</h3>

<ul class="return-block"><li>    (train_data, test_data)</li></ul>

