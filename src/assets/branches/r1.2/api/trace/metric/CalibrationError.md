## CalibrationError<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/trace/metric/calibration_error.py/#L27-L96>View source on Github</a>
```python
CalibrationError(
	true_key: str,
	pred_key: str,
	mode: Union[str, Set[str]]=('eval', 'test'),
	output_name: str='calibration_error',
	method: str='marginal',
	confidence_interval: Union[int, NoneType]=None
)
```
A trace which computes the calibration error for a given set of predictions.

Unlike many common calibration error estimation algorithms, this one has actual theoretical bounds on the quality
of its output: https://arxiv.org/pdf/1909.10155v1.pdf.


<h3>Args:</h3>


* **true_key**: Name of the key that corresponds to ground truth in the batch dictionary.

* **pred_key**: Name of the key that corresponds to predicted score in the batch dictionary.

* **mode**: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **output_name**: What to call the output from this trace (for example in the logger output).

* **method**: Either 'marginal' or 'top-label'. 'marginal' calibration averages the calibration error over each class, whereas 'top-label' computes the error based on only the most confident predictions.

* **confidence_interval**: The calibration error confidence interval to be reported (estimated empirically). Should be in the range (0, 100), or else None to omit this extra calculation.

