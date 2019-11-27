

### get_target
```python
get_target(anchorbox, label, x1, y1, width, height)
```
Generates classification and localization ground-truths.

#### Args:

* **anchorbox (array)** :  anchor boxes
* **label (array)** :  labels for each anchor box.
* **x1 (array)** :  x-coordinate of top left point of the box.
* **y1 (array)** :  y-coordinate of top left point of the box.
* **width (array)** :  width of the box.
* **height (array)** :  height of the box.

#### Returns:

* **array** :  classification groundtruths for each anchor box.
* **array** :  localization groundtruths for each anchor box.