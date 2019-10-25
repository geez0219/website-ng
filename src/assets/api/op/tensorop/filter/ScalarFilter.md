## ScalarFilter
```python
ScalarFilter(inputs, filter_value, keep_prob, mode='train')
```
Class for performing filtering on dataset based on scalar values.

#### Args:

* **inputs** :  Name of the key in the dataset that is to be filtered.
* **filter_value** :  The values in the dataset that are to be filtered.
* **keep_prob** :  The probability of keeping the example.
* **mode** :  mode that the filter acts on.

### forward
```python
forward(self, data, state)
```
Filters the data based on the scalar filter_value.

#### Args:

* **data** :  Data to be filtered.
* **state** :  Information about the current execution context.

#### Returns:
            Tensor containing filtered data.        