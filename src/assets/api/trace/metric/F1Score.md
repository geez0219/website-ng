## F1Score
```python
F1Score(true_key, pred_key=None, labels=None, pos_label=1, average='auto', sample_weight=None, mode='eval', output_name='f1score')
```
Calculate F1 score for classification task and report it back to logger.

#### Args:

* **true_key (str)** :  Name of the keys in the ground truth label in data pipeline.
* **pred_key (str, optional)** :  Name of the keys in predicted label. Default is `None`.
* **labels (list, optional)** :  The set of labels to include. For more details, please refer to        sklearn.netrics.f1_score. Defaults to None.
* **pos_label (str or int, optional)** :  The class to report. For more details, please refer to        sklearn.netrics.f1_score. Defaults to 1.
* **average (str, optional)** :  It should be one of {"auto", "binary", "micro", "macro", "weighted", "samples", None}.        If "auto", the trace will detect the input data type and choose the right average for you. Otherwise, it        will pass its to sklearn.metrics.f1_score. Defaults to "auto".
* **sample_weight (array-like of shape, optional)** :  Sample weights, For more details, please refer to        sklearn.netrics.f1_score. Defaults to None.
* **mode (str, optional)** :  Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always                execute. Defaults to 'eval'.
* **output_name (str, optional)** :  Name of the key to store back to the state. Defaults to "f1score".