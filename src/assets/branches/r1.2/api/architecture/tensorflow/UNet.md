## UNet<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/architecture/tensorflow/unet.py/#L23-L82>View source on Github</a>
```python
UNet(
	input_size: Tuple[int, int, int]=(128, 128, 1)
)
-> tensorflow.python.keras.engine.training.Model
```
A standard UNet implementation in pytorch


<h3>Args:</h3>


* **input_size**: The size of the input tensor (height, width, channels). 

<h3>Raises:</h3>


* **ValueError**: Length of `input_size` is not 3.

* **ValueError**: `input_size`[0] or `input_size`[1] is not a multiple of 16.



<h3>Returns:</h3>

<ul class="return-block"><li>    A TensorFlow LeNet model.</li></ul>

