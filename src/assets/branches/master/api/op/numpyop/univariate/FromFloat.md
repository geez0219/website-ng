## FromFloat
```python
FromFloat(*args, **kwargs)
```
Takes an input float image in range [0, 1.0] and then multiplies by `max_value` to get an int image.


#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **max_value** :  The maximum value to serve as the multiplier. If None it will be inferred by dtype.
* **dtype** :  The data type to cast the output as.
* **Image types** :     float32