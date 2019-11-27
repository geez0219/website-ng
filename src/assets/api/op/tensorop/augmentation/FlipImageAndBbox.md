## FlipImageAndBbox
```python
FlipImageAndBbox(inputs=None, outputs=None, mode=None, flip_left_right=False, flip_up_down=False)
```
This clas flips, at 0.5 probability, image and its associated bounding boxes. The bounding box format is    [x1, y1, width, height].

#### Args:

* **flip_left_right** :  Boolean representing whether to flip the image horizontally with a probability of 0.5. Default            is True.
* **flip_up_down** :  Boolean representing whether to flip the image vertically with a probability of 0.5. Defult is            Flase.
* **mode** :  Augmentation on 'training' data or 'evaluation' data.    

### forward
```python
forward(self, data, state)
```
Transforms the data with the augmentation transformation

#### Args:

* **data** :  Data to be transformed
* **state** :  Information about the current execution context

#### Returns:
            Flipped data.        