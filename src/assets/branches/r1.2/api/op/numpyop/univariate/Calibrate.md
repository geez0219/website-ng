## Calibrate<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/univariate/calibate.py/#L31-L73>View source on Github</a>
```python
Calibrate(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	calibration_fn: Union[str, Callable[[numpy.ndarray], numpy.ndarray]],
	mode: Union[NoneType, str, Iterable[str]]=('test', 'infer')
)
```
Calibrate model predictions using a given calibration function.

This is often used in conjunction with the PBMCalibrator trace. It should be placed in the fe.Network postprocessing
op list.


<h3>Args:</h3>


* **inputs**: Key(s) of predictions to be calibrated.

* **outputs**: Key(s) into which to write the calibrated predictions.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **calibration_fn**: The path to a dill-pickled calibration function, or an in-memory calibration function to apply. If a path is provided, it will be lazy-loaded and so the saved file does not need to exist already when training begins.

