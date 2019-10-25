## Resize
```python
Resize(size, resize_method='bilinear', inputs=None, outputs=None, mode=None)
```
Preprocessing class for resizing the images.

#### Args:

* **size** :  Destination shape of the images.
* **resize_method** :  One of resize methods provided by tensorflow to be used.
* **inputs** :  Name of the key in the dataset that is to be filtered.
* **outputs** :  Name of the key to be created/used in the dataset to store the results.
* **mode** :  mode that the filter acts on.

### forward
```python
forward(self, data, state)
```
Resizes data tensor.

#### Args:

* **data** :  Tensor to be resized.
* **state** :  Information about the current execution context.

#### Returns:
            Resized tensor.        