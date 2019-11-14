

### load_image
```python
load_image(file_path, strip_alpha=False, channels=3)
```


#### Args:

* **file_path** :  The path to an image file
* **strip_alpha** :  True to convert an RGBA image to RGB
* **channels** :  How many channels should the image have (0,1,3)

#### Returns:
    The image loaded into memory and scaled to a range of [-1, 1]