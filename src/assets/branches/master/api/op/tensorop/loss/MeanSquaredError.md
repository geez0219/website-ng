## MeanSquaredError
```python
MeanSquaredError(*args, **kwargs)
```
Calculate the mean squared error loss between two tensors.


#### Args:

* **inputs** :  A tuple or list like [<y_pred>, <y_true>].
* **outputs** :  String key under which to store the computed loss.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **average_loss** :  Whether to average the element-wise loss after the Loss Op.