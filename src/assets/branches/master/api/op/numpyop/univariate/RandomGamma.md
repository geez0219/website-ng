## RandomGamma
```python
RandomGamma(*args, **kwargs)
```
Apply a gamma transform to an image.


#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **gamma_limit** :  If gamma_limit is a single float value, the range will be (-gamma_limit, gamma_limit).
* **eps** :  A numerical stability constant to avoid division by zero.
* **Image types** :     uint8, float32