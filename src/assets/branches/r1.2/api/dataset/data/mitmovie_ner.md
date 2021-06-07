# tf.dataset.data.mitmovie_ner<span class="tag">module</span>

---

## get_sentences_and_labels<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/dataset/data/mitmovie_ner.py/#L30-L59>View source on Github</a>
```python
get_sentences_and_labels(
	path: str
)
-> Tuple[List[str], List[List[str]], Set[str], Set[str]]
```
Combines tokens into sentences and create vocab set for train data and labels.

For simplicity tokens with 'O' entity are omitted.


<h3>Args:</h3>


* **path**: Path to the downloaded dataset file. 

<h3>Returns:</h3>

<ul class="return-block"><li>    (sentences, labels, train_vocab, label_vocab)</li></ul>

---

## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/dataset/data/mitmovie_ner.py/#L62-L105>View source on Github</a>
```python
load_data(
	root_dir: Union[str, NoneType]=None
)
-> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset, Set[str], Set[str]]
```
Load and return the MIT Movie dataset.

MIT Movies dataset is a semantically tagged training and test corpus in BIO format. The sentence is encoded as one
token per line with information provided in tab-seprated columns.
Sourced from https://groups.csail.mit.edu/sls/downloads/movie/


<h3>Args:</h3>


* **root_dir**: The path to store the downloaded data. When `path` is not provided, the data will be saved into `fastestimator_data` under the user's home directory. 

<h3>Returns:</h3>

<ul class="return-block"><li>    (train_data, eval_data, train_vocab, label_vocab)</li></ul>

