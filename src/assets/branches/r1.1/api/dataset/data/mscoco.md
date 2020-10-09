# tf.dataset.data.mscoco<span class="tag">module</span>

---

## MSCOCODataset<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/data/mscoco.py/#L34-L157>View source on Github</a>
```python
MSCOCODataset(
	image_dir: str,
	annotation_file: str,
	caption_file: str,
	include_bboxes: bool=True,
	include_masks: bool=False,
	include_captions: bool=False, min_bbox_area=1.0
)
-> None
```
A specialized DirDataset to handle MSCOCO data.

This dataset combines images from the MSCOCO data directory with their corresponding bboxes, masks, and captions.


<h3>Args:</h3>


* **image_dir**: The path the directory containing MSOCO images.

* **annotation_file**: The path to the file containing annotation data.

* **caption_file**: The path the file containing caption data.

* **include_bboxes**: Whether images should be paired with their associated bounding boxes. If true, images without bounding boxes will be ignored and other images may be oversampled in order to take their place.

* **include_masks**: Whether images should be paired with their associated masks. If true, images without masks will be ignored and other images may be oversampled in order to take their place.

* **include_captions**: Whether images should be paired with their associated captions. If true, images without captions will be ignored and other images may be oversampled in order to take their place.

* **min_bbox_area**: Bounding boxes with a total area less than `min_bbox_area` will be discarded.

---

## load_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/dataset/data/mscoco.py/#L160-L220>View source on Github</a>
```python
load_data(
	root_dir: Union[str, NoneType]=None,
	load_bboxes: bool=True,
	load_masks: bool=False,
	load_captions: bool=False
)
-> Tuple[fastestimator.dataset.data.mscoco.MSCOCODataset, fastestimator.dataset.data.mscoco.MSCOCODataset]
```
Load and return the COCO dataset.


<h3>Args:</h3>


* **root_dir**: The path to store the downloaded data. When `path` is not provided, the data will be saved into `fastestimator_data` under the user's home directory.

* **load_bboxes**: Whether to load bbox-related data.

* **load_masks**: Whether to load mask data (in the form of an array of 1-hot images).

* **load_captions**: Whether to load caption-related data. 

<h3>Returns:</h3>

<ul class="return-block"><li>    (train_data, eval_data)</li></ul>

