## MSCOCODataset
```python
MSCOCODataset(image_dir:str, annotation_file:str, caption_file:str, include_bboxes:bool=True, include_masks:bool=False, include_captions:bool=False, min_bbox_area=1.0) -> None
```
A specialized DirDataset to handle MSCOCO data.    This dataset combines images from the MSCOCO data directory with their corresponding bboxes, masks, and captions.

#### Args:

* **image_dir** :  The path the directory containing MSOCO images.
* **annotation_file** :  The path to the file containing annotation data.
* **caption_file** :  The path the file containing caption data.
* **include_bboxes** :  Whether images should be paired with their associated bounding boxes. If true, images without            bounding boxes will be ignored and other images may be oversampled in order to take their place.
* **include_masks** :  Whether images should be paired with their associated masks. If true, images without masks will            be ignored and other images may be oversampled in order to take their place.
* **include_captions** :  Whether images should be paired with their associated captions. If true, images without            captions will be ignored and other images may be oversampled in order to take their place.
* **min_bbox_area** :  Bounding boxes with a total area less than `min_bbox_area` will be discarded.    