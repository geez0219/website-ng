## ConfusionMatrix
```python
ConfusionMatrix(true_key, pred_key, num_classes, mode='eval', output_name='confusion_matrix')
```
Computes confusion matrix between y_true and y_predict.

#### Args:

* **true_key (str)** :  Name of the key that corresponds to ground truth in batch dictionary
* **pred_key (str)** :  Name of the key that corresponds to predicted score in batch dictionary
* **num_classes (int)** :  Total number of classes of the confusion matrix.
* **mode (str, optional)** :  Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always                execute. Defaults to 'eval'.
* **output_name (str, optional)** :  Name of the key to store to the state. Defaults to "confusion_matrix".