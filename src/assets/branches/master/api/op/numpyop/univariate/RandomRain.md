## RandomRain
```python
RandomRain(*args, **kwargs)
```
Add rain to an image


#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **slant_lower** :  Should be in range [-20, 20].
* **slant_upper** :  Should be in range [-20, 20].
* **drop_length** :  Should be in range [0, 100].
* **drop_width** :  Should be in range [1, 5].
* **drop_color** :  Rain lines color (r, g, b).
* **blur_value** :  How blurry to make the rain.
* **brightness_coefficient** :  Rainy days are usually shady. Should be in range [0, 1].
* **rain_type** :  One of [None, "drizzle", "heavy", "torrential"].
* **Image types** :     uint8, float32