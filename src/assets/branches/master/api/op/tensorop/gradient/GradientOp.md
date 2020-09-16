## GradientOp
```python
GradientOp(*args, **kwargs)
```
Return the gradients of finals w.r.t. inputs.


#### Args:

* **inputs** :  The tensor(s) to compute gradients with respect to.
* **finals** :  The tensor(s) to compute gradients from.
* **outputs** :  The key(s) under which to save the gradients.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".