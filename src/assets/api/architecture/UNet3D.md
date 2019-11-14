

### UNet3D
```python
UNet3D(input_size=(9, 512, 512, 1), clip=(1024, 2048), dropout=0.5, nchannels=(32, 64, 128, 256), nclasses=1, bn=None, activation=<function <lambda> at 0x7f4d50021c80>, upsampling='copy', dilation_rates=(1, 1, 1, 1), residual=False)
```
Creates a U-Net model.This 3D U-Net model is composed of len(nchannels) "contracting blocks" and len(nchannels) "expansive blocks".

#### Args:

* **input_size (tuple, optional)** :  Shape of input image. Defaults to (9, 512, 512, 1).
* **clip** :  If not None, clips input values between clip[0] and clip[1]
* **dropout** :  If None, applies no dropout; Otherwise, applies dropout of probability equal             to the parameter value (0-1 only)
* **nchannels** :  Number of channels for each conv block; len(nchannels) decides number of blocks
* **nclasses** :  Number of target classes for segmentation
* **bn** :  [None, before, after] adds batchnorm layers across every convolution,        before indicates adding BN before activation function is applied        after indicates adding BN after activation function is applied
* **Check https** : //github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md for related ablations!
* **activation** :  Standard Keras activation functions
* **upsampling** :  (copy, conv) Use copy for interpolation upsampling                     Use conv for transposed convolution based upsampling (learnable)
* **dilation_rates** :  Add dilation to the encoder block conv layers [len(dilation_rates) == len(nchannels)]
* **residual** :  False = no residual connections, True = residual connections in every layer
* **NOTE** :  This particular model squashes down k 3D frames (batch * k * m * m * 1) into          1 output frame (batch * 1 * m * m * nclasses).          If different behavior is desired, please change the CNN model as necessary.

#### Returns:

* **'Model' object** :  U-Net model.