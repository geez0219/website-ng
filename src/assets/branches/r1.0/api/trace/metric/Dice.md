## Dice
```python
Dice(true_key:str, pred_key:str, threshold:float=0.5, mode:Union[NoneType, str, List[str]]=('eval', 'test'), output_name:str='Dice') -> None
```
Dice score for binary classification between y_true and y_predicted.

#### Args:

* **true_key** :  The key of the ground truth mask.
* **pred_key** :  The key of the prediction values.
* **threshold** :  The threshold for binarizing the prediction.
* **mode** :  What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **output_name** :  What to call the output from this trace (for example in the logger output).    