## MCC<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/trace/metric/mcc.py/#L27-L101>View source on Github</a>
```python
MCC(
	true_key: str,
	pred_key: str,
	mode: Union[str, Set[str]]=('eval', 'test'),
	output_name: str='mcc', **kwargs
)
-> None
```
A trace which computes the Matthews Correlation Coefficient for a given set of predictions.

This is a preferable metric to accuracy or F1 score since it automatically corrects for class imbalances and does
not depend on the choice of target class (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6941312/). Ideal value is 1,
 a value of 0 means your predictions are completely uncorrelated with the true data. A value less than zero implies
anti-correlation (you should invert your classifier predictions in order to do better).


<h3>Args:</h3>


* **true_key**: Name of the key that corresponds to ground truth in the batch dictionary.

* **pred_key**: Name of the key that corresponds to predicted score in the batch dictionary.

* **mode**: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **output_name**: What to call the output from this trace (for example in the logger output).

* ****kwargs**: Additional keyword arguments that pass to sklearn.metrics.matthews_corrcoef() 

<h3>Raises:</h3>


* **ValueError**: One of ["y_true", "y_pred"] argument exists in `kwargs`.

---

### check_kwargs<span class="tag">method of MCC</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/trace/metric/mcc.py/#L86-L101>View source on Github</a>
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


* **ValueError**: One of ["y_true", "y_pred"] argument exists in `kwargs`.

