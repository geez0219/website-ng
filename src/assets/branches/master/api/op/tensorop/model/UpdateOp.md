## UpdateOp
```python
UpdateOp(*args, **kwargs)
```
This class performs updates to a model's weights based on the loss.


#### Args:

* **model** :  Model instance compiled by fe.build.
* **loss_name** :  The name of loss.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".