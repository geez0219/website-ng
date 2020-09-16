## Normalize
```python
Normalize(*args, **kwargs)
```
Divide pixel values by a maximum value, subtract mean per channel and divide by std per channel.


#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **mean** :  Mean values to subtract.
* **std** :  The divisor.
* **max_pixel_value** :  Maximum possible pixel value.
* **Image types** :     uint8, float32