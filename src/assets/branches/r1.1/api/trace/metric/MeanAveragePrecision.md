## MeanAveragePrecision
```python
MeanAveragePrecision(
	num_classes: int, true_key='bbox',
	pred_key: str='pred',
	mode: str='eval', output_name=('mAP', 'AP50', 'AP75')
)
-> None
```
Calculate COCO mean average precision.

The value of 'y_pred' has shape [batch, num_box, 7] where 7 is [x1, y1, w, h, label, label_score, select], select
is either 0 or 1.
The value of 'bbox' has shape (batch_size, num_bbox, 5). The 5 is [x1, y1, w, h, label].


#### Args:

* **num_classes** :  Maximum `int` value for your class label. In COCO dataset we only used 80 classes, but the maxium        value of the class label is `90`. In this case `num_classes` should be `90`.

#### Returns:
    Mean Average Precision.

### accumulate
```python
accumulate(
	self
)
-> None
```
Generate precision-recall curve.

### compute_iou
```python
compute_iou(
	self,
	det: numpy.ndarray,
	gt: numpy.ndarray
)
-> numpy.ndarray
```
Compute intersection over union.

We leverage `maskUtils.iou`.


#### Args:

* **det** :  Detection array.
* **gt** :  Ground truth array.

#### Returns:
    Intersection of union array.

### evaluate_img
```python
evaluate_img(
	self,
	cat_id: int,
	img_id: int
)
-> Dict
```
Find gt matches for det given one image and one category.


#### Args:

* **cat_id** : 
* **img_id** : 

#### Returns:


### on_batch_begin
```python
on_batch_begin(
	self,
	data: fastestimator.util.data.Data
)
```
Reset instance variables.

### on_epoch_begin
```python
on_epoch_begin(
	self,
	data: fastestimator.util.data.Data
)
```
Reset instance variables.

### summarize
```python
summarize(
	self,
	iou: float=None
)
-> float
```
Compute average precision given one intersection union threshold.


#### Args:

* **iou** :  Intersection over union threshold. If this value is `None`, then average all iou thresholds. The result        is the mean average precision.

#### Returns:
    Average precision.