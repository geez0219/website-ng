

### get_loc_offset
```python
get_loc_offset(box_gt, box_anchor)
```
Computes the offset of a groundtruth box and an anchor box.

#### Args:

* **box_gt (array)** :  groundtruth box.
* **box_anchor (array)** :  anchor box.

#### Returns:

* **float** :  offset between x1 coordinate of the two boxes.
* **float** :  offset between y1 coordinate of the two boxes.
* **float** :  offset between x2 coordinate of the two boxes.
* **float** :  offset between y2 coordinate of the two boxes.