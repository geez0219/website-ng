

### tensor_pow
```python
tensor_pow(tensor:~Tensor, power:Union[int, float]) -> ~Tensor
```
Computes x^power element-wise along `tensor`.

This method can be used with Numpy data:
```python
n = np.array([[1, 4, 6], [2.3, 0.5, 0]])
b = fe.backend.tensor_pow(n, 3.2)  # [[1.0, 84.449, 309.089], [14.372, 0.109, 0]]
b = fe.backend.tensor_pow(n, 0.21)  # [[1.0, 1.338, 1.457], [1.191, 0.865, 0]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[1, 4, 6], [2.3, 0.5, 0]])
b = fe.backend.tensor_pow(t, 3.2)  # [[1.0, 84.449, 309.089], [14.372, 0.109, 0]]
b = fe.backend.tensor_pow(t, 0.21)  # [[1.0, 1.338, 1.457], [1.191, 0.865, 0]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[1, 4, 6], [2.3, 0.5, 0]])
b = fe.backend.tensor_pow(p, 3.2)  # [[1.0, 84.449, 309.089], [14.372, 0.109, 0]]
b = fe.backend.tensor_pow(p, 0.21)  # [[1.0, 1.338, 1.457], [1.191, 0.865, 0]]
```


#### Args:

* **tensor** :  The input tensor.
* **power** :  The power to which to raise the elements in the `tensor`.

#### Returns:
    The `tensor` raised element-wise to the given `power`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.