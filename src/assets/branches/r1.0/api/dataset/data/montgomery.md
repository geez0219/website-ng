# fastestimator.dataset.data.montgomery<span class="tag">module</span>
---
## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/dataset/data/montgomery.py/#L30-L79>View source on Github</a>
```python
load_data(
	root_dir: Union[str, NoneType]=None
)
-> fastestimator.dataset.csv_dataset.CSVDataset
```
Load and return the montgomery dataset.

Sourced from http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip. This method will download the data
    to local storage if the data has not been previously downloaded.


<h3>Args:</h3>

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.

<h3>Returns:</h3>
    train_data

