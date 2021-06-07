# tf.dataset.data.imdb_review<span class="tag">module</span>

---

## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/dataset/data/imdb_review.py/#L37-L56>View source on Github</a>
```python
load_data(
	max_len: int,
	vocab_size: int
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset]
```
Load and return the IMDB Movie review dataset.

This dataset contains 25,000 reviews labeled by sentiments (either positive or negative).


<h3>Args:</h3>


* **max_len**: Maximum desired length of an input sequence.

* **vocab_size**: Vocabulary size to learn word embeddings. 

<h3>Returns:</h3>

<ul class="return-block"><li>    (train_data, eval_data)</li></ul>

---

## pad<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/dataset/data/imdb_review.py/#L23-L34>View source on Github</a>
```python
pad(
	input_list: List[int],
	padding_size: int,
	padding_value: int
)
-> List[int]
```
Pad an input_list to a given size.


<h3>Args:</h3>


* **input_list**: The list to be padded.

* **padding_size**: The desired length of the returned list.

* **padding_value**: The value to be inserted for padding. 

<h3>Returns:</h3>

<ul class="return-block"><li>    <code>input_list</code> with <code>padding_value</code>s appended until the <code>padding_size</code> is reached.</li></ul>

