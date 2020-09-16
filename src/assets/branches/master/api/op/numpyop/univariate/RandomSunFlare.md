## RandomSunFlare
```python
RandomSunFlare(*args, **kwargs)
```
Add a sun flare to the image.


#### Args:

* **inputs** :  Key(s) of images to be normalized.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **flare_roi** :  region of the image where flare will appear (x_min, y_min, x_max, y_max). All values should be        in range [0, 1].
* **angle_lower** :  should be in range [0, `angle_upper`].
* **angle_upper** :  should be in range [`angle_lower`, 1].
* **num_flare_circles_lower** :  lower limit for the number of flare circles.        Should be in range [0, `num_flare_circles_upper`].
* **num_flare_circles_upper** :  upper limit for the number of flare circles.        Should be in range [`num_flare_circles_lower`, inf].
* **src_radius** :  Radius of the flare.
* **src_color** :  Color of the flare (R,G,B).
* **Image types** :     uint8, float32