## MeanAveragePrecision<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/metric/mean_average_precision.py/#L27-L424>View source on Github</a>
```python
MeanAveragePrecision(
	num_classes: int, true_key='bbox',
	pred_key: str='pred',
	mode: str='eval', output_name=('mAP', 'AP50', 'AP75')
)
-> None
```
Calculate COCO mean average precision.


<h3>Args:</h3>


* **num_classes**: Maximum `int` value for your class label. In COCO dataset we only used 80 classes, but the maxium value of the class label is `90`. In this case `num_classes` should be `90`. 

<h3>Returns:</h3>

<ul class="return-block"><li>    Mean Average Precision.</li></ul>

---

### accumulate<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/metric/mean_average_precision.py/#L276-L367>View source on Github</a>
```python
accumulate(
	self
)
-> None
```
Generate precision-recall curve.

---

### compute_iou<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/metric/mean_average_precision.py/#L393-L424>View source on Github</a>
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

### evaluate_img<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/metric/mean_average_precision.py/#L214-L274>View source on Github</a>
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

### on_batch_begin<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/metric/mean_average_precision.py/#L104-L112>View source on Github</a>
```python
on_batch_begin(
	self,
	data: fastestimator.util.data.Data
)
```
Reset instance variables.

---

### on_epoch_begin<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/metric/mean_average_precision.py/#L97-L102>View source on Github</a>
```python
on_epoch_begin(
	self,
	data: fastestimator.util.data.Data
)
```
Reset instance variables.

---

### summarize<span class="tag">method of MeanAveragePrecision</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/metric/mean_average_precision.py/#L369-L391>View source on Github</a>
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

