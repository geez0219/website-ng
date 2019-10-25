## Dice
```python
Dice(true_key, pred_key, threshold=0.5, mode='eval', output_name='dice')
```
Computes Dice score for binary classification between y_true and y_predict.

#### Args:

* **true_key (str)** :  Name of the keys in the ground truth label in data pipeline.
* **pred_key (str, optional)** :  Mame of the keys in predicted label. Default is `None`.
* **threshold (float, optional)** :  Threshold of the prediction. Defaults to 0.5.
* **mode (str, optional)** :  Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always                execute. Defaults to 'eval'.
* **output_name (str, optional)** :  Name of the key to store to the state. Defaults to "dice".