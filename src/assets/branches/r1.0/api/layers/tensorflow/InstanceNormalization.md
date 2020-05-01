## InstanceNormalization
```python
InstanceNormalization(epsilon:float=1e-05) -> None
```
A layer for performing instance normalization.    This layer assumes that you are using the a tensor shaped like (Batch, Height, Width, Channels). See
* **https** : //arxiv.org/abs/1607.08022 for details about this layer. The implementation here is borrowed from
* **https** : //github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py.    ```python    n = tfp.distributions.Normal(loc=10, scale=2)    x = n.sample(sample_shape=(1, 100, 100, 1))  # mean ~= 10, stddev ~= 2    m = fe.layers.tensorflow.InstanceNormalization()    y = m(x)  # mean ~= 0, stddev ~= 0    ```

#### Args:

* **epsilon** :  A numerical stability constant added to the variance.    