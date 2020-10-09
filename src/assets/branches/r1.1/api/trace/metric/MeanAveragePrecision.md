## MeanAveragePrecision<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/metric/mean_average_precision.py/#L29-L430>View source on Github</a>
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


<h3>Args:</h3>


* **num_classes**: Maximum `int` value for your class label. In COCO dataset we only used 80 classes, but the maxium value of the class label is `90`. In this case `num_classes` should be `90`. 

<h3>Returns:</h3>

<ul class="return-block"><li>    Mean Average Precision.</li></ul>

---

### accumulate<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/metric/mean_average_precision.py/#L282-L373>View source on Github</a>
```python
accumulate(
	self
)
-> None
```
Generate precision-recall curve.

---

### compute_iou<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/metric/mean_average_precision.py/#L399-L430>View source on Github</a>
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


<h4>Args:</h4>


* **det**: Detection array.

* **gt**: Ground truth array. 

<h4>Returns:</h4>

<ul class="return-block"><li>    Intersection of union array.</li></ul>

---

### evaluate_img<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/metric/mean_average_precision.py/#L220-L280>View source on Github</a>
```python
evaluate_img(
	self,
	cat_id: int,
	img_id: int
)
-> Dict
```
Find gt matches for det given one image and one category.


<h4>Args:</h4>

 cat_id: img_id: Returns:

---

### on_batch_begin<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/metric/mean_average_precision.py/#L110-L118>View source on Github</a>
```python
on_batch_begin(
	self,
	data: fastestimator.util.data.Data
)
```
Reset instance variables.

---

### on_epoch_begin<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/metric/mean_average_precision.py/#L103-L108>View source on Github</a>
```python
on_epoch_begin(
	self,
	data: fastestimator.util.data.Data
)
```
Reset instance variables.

---

### summarize<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/metric/mean_average_precision.py/#L375-L397>View source on Github</a>
```python
summarize(
	self,
	iou: float=None
)
-> float
```
Compute average precision given one intersection union threshold.


<h4>Args:</h4>


* **iou**: Intersection over union threshold. If this value is `None`, then average all iou thresholds. The result is the mean average precision. 

<h4>Returns:</h4>

<ul class="return-block"><li>    Average precision.</li></ul>

