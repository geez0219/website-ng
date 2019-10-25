## SubPixelConv2D
```python
SubPixelConv2D(upsample_factor=2, nchannels=128)
```
Class for upsampling using subpixel convolution (https://arxiv.org/pdf/1609.05158.pdf)

#### Args:

* **upsample_factor (int, optional)** :  [description]. Defaults to 2.
* **nchannels (int, optional)** :  [description]. Defaults to 128.

### compute_output_shape
```python
compute_output_shape(self, s)
```
 If you are using "channels_last" configuration

### get_config
```python
get_config(self)
```
Get JSON config for params

#### Returns:

* **[dict]** :  params defining subpixel convolution layer        