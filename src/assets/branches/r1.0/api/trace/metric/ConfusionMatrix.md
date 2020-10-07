## ConfusionMatrix<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/metric/confusion_matrix.py/#L25-L76>View source on Github</a>
```python
ConfusionMatrix(
	true_key: str,
	pred_key: str,
	num_classes: int,
	mode: Union[str, Set[str]]=('eval', 'test'),
	output_name: str='confusion_matrix'
)
-> None
```
Computes the confusion matrix between y_true and y_predicted.


<h3>Args:</h3>

* **true_key** :  Name of the key that corresponds to ground truth in the batch dictionary.
* **pred_key** :  Name of the key that corresponds to predicted score in the batch dictionary.
* **num_classes** :  Total number of classes of the confusion matrix.
* **mode** :  What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **output_name** :  Name of the key to store to the state.



