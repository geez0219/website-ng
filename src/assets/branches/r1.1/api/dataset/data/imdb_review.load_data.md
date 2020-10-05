

### imdb_review.load_data
```python
imdb_review.load_data(
	max_len: int,
	vocab_size: int
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset]
```
Load and return the IMDB Movie review dataset.

This dataset contains 25,000 reviews labeled by sentiments (either positive or negative).


#### Args:

* **max_len** :  Maximum desired length of an input sequence.
* **vocab_size** :  Vocabulary size to learn word embeddings.

#### Returns:
    (train_data, eval_data)