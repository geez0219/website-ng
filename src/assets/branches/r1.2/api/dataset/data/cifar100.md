# tf.dataset.data.cifar100<span class="tag">module</span>

---

## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/dataset/data/cifar100.py/#L22-L46>View source on Github</a>
```python
load_data(
	image_key: str='x',
	label_key: str='y',
	label_mode: str='fine'
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset]
```
Load and return the CIFAR100 dataset.

Please consider using the ciFAIR100 dataset instead. CIFAR100 contains duplicates between its train and test sets.


<h3>Args:</h3>


* **image_key**: The key for image.

* **label_key**: The key for label.

* **label_mode**: Either "fine" for 100 classes or "coarse" for 20 classes. 

<h3>Raises:</h3>


* **ValueError**: If the label_mode is invalid.

<h3>Returns:</h3>

<ul class="return-block"><li>    (train_data, eval_data)

</li></ul>

