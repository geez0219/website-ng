## GeneratorDataset<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/dataset/generator_dataset.py/#L23-L79>View source on Github</a>
```python
GeneratorDataset(
	generator: Generator[Dict[str, Any], int, NoneType],
	samples_per_epoch: int
)
-> None
```
A dataset from a generator function.


<h3>Args:</h3>

* **generator** :  The generator function to invoke in order to get a data sample.
* **samples_per_epoch** :  How many samples should be drawn from the generator during an epoch. Note that the generator        function will actually be invoke more times than the number specified here due to backend validation        routines.

### summary<span class="tag">method of GeneratorDataset</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/dataset/generator_dataset.py/#L66-L79>View source on Github</a>
```python
summary(
	self
)
-> fastestimator.dataset.dataset.DatasetSummary
```
Generate a summary representation of this dataset.

<h4>Returns:</h4>
    A summary representation of this dataset.



