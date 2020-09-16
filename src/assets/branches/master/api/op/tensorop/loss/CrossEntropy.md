## CrossEntropy
```python
CrossEntropy(*args, **kwargs)
```
Calculate Element-Wise CrossEntropy (binary, categorical or sparse categorical).


#### Args:

* **inputs** :  A tuple or list like [<y_pred>, <y_true>].
* **outputs** :  String key under which to store the computed loss value.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **from_logits** :  Whether y_pred is logits (without softmax).
* **average_loss** :  Whether to average the element-wise loss after the Loss Op.
* **form** :  What form of cross entropy should be performed ('binary', 'categorical', 'sparse', or None). None will        automatically infer the correct form based on tensor shape.