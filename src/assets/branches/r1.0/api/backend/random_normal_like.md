

### random_normal_like
```python
random_normal_like(tensor:~Tensor, mean:float=0.0, std:float=1.0, dtype:Union[NoneType, str]='float32') -> ~Tensor
```
Generate noise shaped like `tensor` from a random normal distribution with a given `mean` and `std`.
* **This method can be used with Numpy data** : ```pythonn = np.array([[0,1],[2,3]])b = fe.backend.random_normal_like(n)  # [[-0.6, 0.2], [1.9, -0.02]]b = fe.backend.random_normal_like(n, mean=5.0)  # [[3.7, 5.7], [5.6, 3.6]]```
* **This method can be used with TensorFlow tensors** : ```pythont = tf.constant([[0,1],[2,3]])b = fe.backend.random_normal_like(t)  # [[-0.6, 0.2], [1.9, -0.02]]b = fe.backend.random_normal_like(t, mean=5.0)  # [[3.7, 5.7], [5.6, 3.6]]```
* **This method can be used with PyTorch tensors** : ```pythonp = torch.tensor([[0,1],[2,3]])b = fe.backend.random_normal_like(p)  # [[-0.6, 0.2], [1.9, -0.02]]b = fe.backend.random_normal_like(P, mean=5.0)  # [[3.7, 5.7], [5.6, 3.6]]```

#### Args:

* **tensor** :  The tensor whose shape will be copied.
* **mean** :  The mean of the normal distribution to be sampled.
* **std** :  The standard deviation of the normal distribution to be sampled.
* **dtype** :  The data type to be used when generating the resulting tensor. This should be one of the floating point        types.

#### Returns:
    A tensor of random normal noise with the same shape as `tensor`.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.