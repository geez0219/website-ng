## Accuracy
```python
Accuracy(true_key:str, pred_key:str, mode:Union[str, Set[str]]=('eval', 'test'), output_name:str='accuracy') -> None
```
A trace which computes the accuracy for a given set of predictions.
* **Consider using MCC instead** :  https//www.ncbi.nlm.nih.gov/pmc/articles/PMC6941312/

#### Args:

* **true_key** :  Name of the key that corresponds to ground truth in the batch dictionary.
* **pred_key** :  Name of the key that corresponds to predicted score in the batch dictionary.
* **mode** :  What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **output_name** :  What to call the output from this trace (for example in the logger output).    