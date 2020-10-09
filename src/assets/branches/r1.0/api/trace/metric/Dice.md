## Dice<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/metric/dice.py/#L24-L71>View source on Github</a>
```python
Dice(
	true_key: str,
	pred_key: str,
	threshold: float=0.5,
	mode: Union[NoneType, str, List[str]]=('eval', 'test'),
	output_name: str='Dice'
)
-> None
```
Dice score for binary classification between y_true and y_predicted.


<h3>Args:</h3>


* **true_key**: The key of the ground truth mask.

* **pred_key**: The key of the prediction values.

* **threshold**: The threshold for binarizing the prediction.

* **mode**: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **output_name**: What to call the output from this trace (for example in the logger output).

