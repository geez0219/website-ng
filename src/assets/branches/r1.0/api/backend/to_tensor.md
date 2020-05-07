

### to_tensor
```python
to_tensor(data:Union[Collection, ~Tensor], target_type:str) -> Union[Collection, ~Tensor]
```
Convert tensors within a collection of `data` to a given `target_type` recursively.

This method can be used with Numpy data:
```python
data = {"x": np.ones((10,15)), "y":[np.ones((4)), np.ones((5, 3))], "z":{"key":np.ones((2,2))}}
t = fe.backend.to_tensor(data, target_type='tf')
# {"x": &lt;tf.Tensor&gt;, "y":[&lt;tf.Tensor&gt;, &lt;tf.Tensor&gt;], "z": {"key": &lt;tf.Tensor&gt;}}
p = fe.backend.to_tensor(data, target_type='torch')
# {"x": &lt;torch.Tensor&gt;, "y":[&lt;torch.Tensor&gt;, &lt;torch.Tensor&gt;], "z": {"key": &lt;torch.Tensor&gt;}}
```

This method can be used with TensorFlow tensors:
```python
data = {"x": tf.ones((10,15)), "y":[tf.ones((4)), tf.ones((5, 3))], "z":{"key":tf.ones((2,2))}}
p = fe.backend.to_tensor(data, target_type='torch')
# {"x": &lt;torch.Tensor&gt;, "y":[&lt;torch.Tensor&gt;, &lt;torch.Tensor&gt;], "z": {"key": &lt;torch.Tensor&gt;}}
```

This method can be used with PyTorch tensors:
```python
data = {"x": torch.ones((10,15)), "y":[torch.ones((4)), torch.ones((5, 3))], "z":{"key":torch.ones((2,2))}}
t = fe.backend.to_tensor(data, target_type='tf')
# {"x": &lt;tf.Tensor&gt;, "y":[&lt;tf.Tensor&gt;, &lt;tf.Tensor&gt;], "z": {"key": &lt;tf.Tensor&gt;}}
```



#### Args:

* **data** :  A tensor or possibly nested collection of tensors.
* **target_type** :  What kind of tensor(s) to create, either "tf" or "torch".

#### Returns:
    A collection with the same structure as `data`, but with any tensors converted to the `target_type`.