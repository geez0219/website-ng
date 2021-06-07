## iwd<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/iwd.py/#L32-L90>View source on Github</a>
```python
iwd(
	tensor: ~Tensor,
	power: float=1.0,
	max_prob: float=0.95,
	pairwise_distance: float=1.0,
	eps: Union[~Tensor, NoneType]=None
)
-> ~Tensor
```
Compute the Inverse Weighted Distance from the given input.

This can be used as an activation function for the final layer of a neural network instead of softmax. For example,
instead of: model.add(layers.Dense(classes, activation='softmax')), you could use:
model.add(layers.Dense(classes, activation=lambda x: iwd(tf.nn.sigmoid(x))))

This method can be used with Numpy data:
```python
n = np.array([[0.5]*5, [0]+[1]*4])
b = fe.backend.iwd(n)  # [[0.2, 0.2, 0.2, 0.2, 0.2], [0.95, 0.0125, 0.0125, 0.0125, 0.0125]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[0.5]*5, [0]+[1]*4])
b = fe.backend.iwd(n)  # [[0.2, 0.2, 0.2, 0.2, 0.2], [0.95, 0.0125, 0.0125, 0.0125, 0.0125]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[0.5]*5, [0]+[1]*4])
b = fe.backend.iwd(n)  # [[0.2, 0.2, 0.2, 0.2, 0.2], [0.95, 0.0125, 0.0125, 0.0125, 0.0125]]
```


<h3>Args:</h3>


* **tensor**: The input value. Should be of shape (Batch, C) where every element in C corresponds to a (non-negative) distance to a target class.

* **power**: The power to raise the inverse distances to. 1.0 results in a fairly intuitive probability output. Larger powers can widen regions of certainty, whereas values between 0 and 1 can widen regions of uncertainty.

* **max_prob**: The maximum probability to assign to a class estimate when it is distance zero away from the target. For numerical stability this must be less than 1.0. We have found that using smaller values like 0.95 can lead to natural adversarial robustness.

* **pairwise_distance**: The distance to any other class when the distance to a target class is zero. For example, if you have a perfect match for class 'a', what distance should be reported to class 'b'. If you have a metric where this isn't constant, just use an approximate expected distance. In that case `max_prob` will only give you approximate control over the true maximum probability.

* **eps**: The numeric stability constant to be used when d approaches zero. If None then it will be computed using `max_prob` and `pairwise_distance`. If not None, then `max_prob` and `pairwise_distance` will be ignored. 

<h3>Returns:</h3>

<ul class="return-block"><li>    A probability distribution of shape (Batch, C) where smaller distances from <code>tensor</code> correspond to larger
    probabilities.</li></ul>

