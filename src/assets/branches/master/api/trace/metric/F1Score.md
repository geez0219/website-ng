## F1Score
```python
F1Score(*args, **kwargs)
```
Calculate the F1 score for a classification task and report it back to the logger.

Consider using MCC instead: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6941312/


#### Args:

* **true_key** :  Name of the key that corresponds to ground truth in the batch dictionary.
* **pred_key** :  Name of the key that corresponds to predicted score in the batch dictionary.
* **mode** :  What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **output_name** :  Name of the key to store back to the state.