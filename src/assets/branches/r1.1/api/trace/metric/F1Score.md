## F1Score<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/metric/f1_score.py/#L27-L80>View source on Github</a>
```python
F1Score(
	true_key: str,
	pred_key: str,
	mode: Union[str, Set[str]]=('eval', 'test'),
	output_name: str='f1_score'
)
-> None
```
Calculate the F1 score for a classification task and report it back to the logger.

Consider using MCC instead: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6941312/


<h3>Args:</h3>


* **true_key**: Name of the key that corresponds to ground truth in the batch dictionary.

* **pred_key**: Name of the key that corresponds to predicted score in the batch dictionary.

* **mode**: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **output_name**: Name of the key to store back to the state.

