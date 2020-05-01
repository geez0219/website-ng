## MCC
```python
MCC(true_key:str, pred_key:str, mode:Union[str, Set[str]]=('eval', 'test'), output_name:str='mcc') -> None
```
A trace which computes the Matthews Correlation Coefficient for a given set of predictions.    This is a preferable metric to accuracy or F1 score since it automatically corrects for class imbalances and does
* **not depend on the choice of target class (https** : //www.ncbi.nlm.nih.gov/pmc/articles/PMC6941312/). Ideal value is 1,     a value of 0 means your predictions are completely uncorrelated with the true data. A value less than zero implies    anti-correlation (you should invert your classifier predictions in order to do better).

#### Args:

* **true_key** :  Name of the key that corresponds to ground truth in the batch dictionary.
* **pred_key** :  Name of the key that corresponds to predicted score in the batch dictionary.
* **mode** :  What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **output_name** :  What to call the output from this trace (for example in the logger output).    