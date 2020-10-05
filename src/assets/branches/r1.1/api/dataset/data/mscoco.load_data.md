

### mscoco.load_data
```python
mscoco.load_data(
	root_dir: Union[str, NoneType]=None,
	load_bboxes: bool=True,
	load_masks: bool=False,
	load_captions: bool=False
)
-> Tuple[dataset.data.mscoco.MSCOCODataset, dataset.data.mscoco.MSCOCODataset]
```
Load and return the COCO dataset.


#### Args:

* **root_dir** :  The path to store the downloaded data. When `path` is not provided, the data will be saved into        `fastestimator_data` under the user's home directory.
* **load_bboxes** :  Whether to load bbox-related data.
* **load_masks** :  Whether to load mask data (in the form of an array of 1-hot images).
* **load_captions** :  Whether to load caption-related data.

#### Returns:
    (train_data, eval_data)