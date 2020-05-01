

### load_data
```python
load_data(image_key:str='x', label_key:str='y') -> Tuple[fastestimator.dataset.numpy_dataset.NumpyDataset, fastestimator.dataset.numpy_dataset.NumpyDataset]
```
Load and return the MNIST dataset.

#### Args:

* **image_key** :  The key for image.
* **label_key** :  The key for label.

#### Returns:
    (train_data, eval_data)