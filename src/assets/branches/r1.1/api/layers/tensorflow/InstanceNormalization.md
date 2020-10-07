## InstanceNormalization<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/layers/tensorflow/instance_norm.py/#L21-L61>View source on Github</a>
```python
InstanceNormalization(
	*args, **kwargs
)
```
A layer for performing instance normalization.

This class is intentionally not @traceable (models and layers are handled by a different process).

This layer assumes that you are using the a tensor shaped like (Batch, Height, Width, Channels). See
https://arxiv.org/abs/1607.08022 for details about this layer. The implementation here is borrowed from
https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py.

```python
n = tfp.distributions.Normal(loc=10, scale=2)
x = n.sample(sample_shape=(1, 100, 100, 1))  # mean ~= 10, stddev ~= 2
m = fe.layers.tensorflow.InstanceNormalization()
y = m(x)  # mean ~= 0, stddev ~= 0
```


<h3>Args:</h3>

* **epsilon** :  A numerical stability constant added to the variance.



