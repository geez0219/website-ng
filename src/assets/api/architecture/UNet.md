

### UNet
```python
UNet(input_size=(128, 128, 3), dropout=0.5, nchannels=(64, 128, 256, 512, 1024), nclasses=1, bn=None, activation='relu', upsampling='bilinear', dilation_rates=(1, 1, 1, 1, 1), residual=False)
```
Creates a U-Net model.This U-Net model is composed of len(nchannels) "contracting blocks" and len(nchannels) "expansive blocks".

#### Args:

* **input_size (tuple, optional)** :  Shape of input image. Defaults to (128, 128, 3).
* **dropout** :  If None, applies no dropout; Otherwise, applies dropout of probability equal             to the parameter value (0-1 only)
* **nchannels** :  Number of channels for each conv block; len(nchannels) decides number of blocks
* **nclasses** :  Number of target classes for segmentation
* **bn** :  [None, before, after] adds batchnorm layers across every convolution,        before indicates adding BN before activation function is applied        after indicates adding BN after activation function is applied
* **Check https** : //github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md for related ablations!
* **activation** :  Standard Keras activation functions
* **upsampling** :  (bilinear, nearest, conv, subpixel)            Use bilinear, nearest (nearest neighbour) for predetermined upsampling            Use conv for transposed convolution based upsampling (learnable)            Use subpixel for SubPixel based upsampling (learnable)
* **dilation_rates** :  Add dilation to the encoder block conv layers [len(dilation_rates) == len(nchannels)]
* **residual** :  False = no residual connections, enc = residual connections in encoder layers only              dec = residual connections in decoder layers only, True = residual connections in every layer

#### Returns:

* **'Model' object** :  U-Net model.