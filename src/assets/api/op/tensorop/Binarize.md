## Binarize
```python
Binarize(threshold, inputs=None, outputs=None, mode=None)
```
Binarize data based on threshold between 0 and 1.

#### Args:

* **threshold** :  Threshold for binarizing.
* **inputs** :  Name of the key in the dataset that is to be filtered.
* **outputs** :  Name of the key to be created/used in the dataset to store the results.
* **mode** :  mode that the filter acts on.

### forward
```python
forward(self, data, state)
```
Transforms the image to binary based on threshold.

#### Args:

* **data** :  Data to be binarized.
* **state** :  Information about the current execution context.

#### Returns:
            Tensor containing binarized data.        