## ImageReader
```python
ImageReader(inputs=None, outputs=None, mode=None, parent_path='', grey_scale=False)
```
Class for reading png or jpg images

#### Args:

* **parent_path (str)** :  Parent path that will be added on given path
* **grey_scale (bool)** :  Boolean to indicate whether or not to read image as grayscale

### forward
```python
forward(self, path, state)
```
Reads numpy array from image path

#### Args:

* **path** :  path of the image
* **state** :  A dictionary containing background information such as 'mode'

#### Returns:
           Image as numpy array        