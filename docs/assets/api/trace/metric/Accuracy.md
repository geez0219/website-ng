## Accuracy
```python
Accuracy(true_key, pred_key, mode='eval', output_name='accuracy')
```
Calculates accuracy for classification task and report it back to logger.

#### Args:

* **true_key (str)** :  Name of the key that corresponds to ground truth in batch dictionary
* **pred_key (str)** :  Name of the key that corresponds to predicted score in batch dictionary
* **mode (str, optional)** :  Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always                execute. Defaults to 'eval'.
* **output_name (str, optional)** :  Name of the key to store to the state. Defaults to "accuracy".