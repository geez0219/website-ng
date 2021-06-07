## Precision<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/trace/metric/precision.py/#L27-L102>View source on Github</a>
```python
Precision(
	true_key: str,
	pred_key: str,
	mode: Union[str, Set[str]]=('eval', 'test'),
	output_name: str='precision', **kwargs
)
-> None
```
Computes precision for a classification task and reports it back to the logger.


<h3>Args:</h3>


* **true_key**: Name of the key that corresponds to ground truth in the batch dictionary.

* **pred_key**: Name of the key that corresponds to predicted score in the batch dictionary.

* **mode**: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **output_name**: Name of the key to store to the state.

* ****kwargs**: Additional keyword arguments that pass to sklearn.metrics.precision_score() 

<h3>Raises:</h3>


* **ValueError**: One of ["y_true", "y_pred", "average"] argument exists in `kwargs`.

---

### check_kwargs<span class="tag">method of Precision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/trace/metric/precision.py/#L87-L102>View source on Github</a>
```python
check_kwargs(
	kwargs: Dict[str, Any]
)
-> None
```
Check if `kwargs` has any blacklist argument and raise an error if it does.


<h4>Args:</h4>


* **kwargs**: Keywork arguments to be examined. 

<h4>Raises:</h4>


* **ValueError**: One of ["y_true", "y_pred", "average"] argument exists in `kwargs`.

