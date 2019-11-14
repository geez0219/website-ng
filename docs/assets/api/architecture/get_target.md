

### get_target
```python
get_target(anchorbox, label, x1, y1, x2, y2, num_classes=10)
```
Generates classification and localization ground-truths.

#### Args:

* **anchorbox (array)** :  anchor boxes
* **label (array)** :  labels for each anchor box.
* **x1 (array)** :  x-coordinate of top left point of the box.
* **y1 (array)** :  y-coordinate of top left point of the box.
* **x2 (array)** :  x-coordinate of bottom right point of the box.
* **y2 (array)** :  x-coordinate of bottom right point of the box.
* **num_classes (int, optional)** :  number of classes. Defaults to 10.

#### Returns:

* **array** :  classification groundtruths for each anchor box.
* **array** :  localization groundtruths for each anchor box.