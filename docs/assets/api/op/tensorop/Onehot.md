## Onehot
```python
Onehot(num_dim, inputs=None, outputs=None, mode=None)
```
Preprocessing class for converting categorical labels to onehot encoding.

#### Args:

* **num_dim** :  Number of dimensions of the labels.
* **inputs** :  Name of the key in the dataset that is to be filtered.
* **outputs** :  Name of the key to be created/used in the dataset to store the results.
* **mode** :  mode that the filter acts on.

### forward
```python
forward(self, data, state)
```
Transforms categorical labels to onehot encodings.

#### Args:

* **data** :  Data to be preprocessed.
* **state** :  Information about the current execution context.

#### Returns:
            Transformed labels.        