## MiniBatchStd
```python
MiniBatchStd(group_size=4)
```
A layer that outputs a concatenation of the input tensor and an average of channel-wise standard deviation of the input tensor

#### Args:

* **group_size (int)** :  a parameter determining size of subgroup to compute the statistics