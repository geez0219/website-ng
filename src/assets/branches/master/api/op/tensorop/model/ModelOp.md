## ModelOp
```python
ModelOp(*args, **kwargs)
```
This class performs forward passes of a neural network over batch data to generate predictions.


#### Args:

* **model** :  A model compiled by fe.build.
* **inputs** :  String key of input training data.
* **outputs** :  String key under which to store predictions.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **trainable** :  Indicates whether the model should have its weights tracked for update.