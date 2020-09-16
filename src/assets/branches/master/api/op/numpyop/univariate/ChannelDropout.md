## ChannelDropout
```python
ChannelDropout(*args, **kwargs)
```
Randomly drop channels from the image.


#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **channel_drop_range** :  Range from which we choose the number of channels to drop.
* **fill_value** :  Pixel values for the dropped channel.
* **Image types** :     int8, uint16, unit32, float32