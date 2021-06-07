# tf.dataset.data.cifair100<span class="tag">module</span>

---

## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/dataset/data/cifair100.py/#L25-L74>View source on Github</a>
```python
load_data(
	image_key: str='x',
	label_key: str='y',
	label_mode: str='fine'
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset]
```
Load and return the ciFAIR100 dataset.

This is the cifar100 dataset but with test set duplicates removed and replaced. See
https://arxiv.org/pdf/1902.00423.pdf or https://cvjena.github.io/cifair/ for details. Cite the paper if you use the
dataset.


<h3>Args:</h3>


* **image_key**: The key for image.

* **label_key**: The key for label.

* **label_mode**: Either "fine" for 100 classes or "coarse" for 20 classes. 

<h3>Raises:</h3>


* **ValueError**: If the label_mode is invalid.

<h3>Returns:</h3>

<ul class="return-block"><li>    (train_data, test_data)

</li></ul>

