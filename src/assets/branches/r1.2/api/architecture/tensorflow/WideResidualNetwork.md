## WideResidualNetwork<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/architecture/tensorflow/wideresnet.py/#L22-L56>View source on Github</a>
```python
WideResidualNetwork(
	input_shape: Tuple[int, int, int],
	depth: int=28,
	widen_factor: int=10,
	dropout: float=0.0,
	classes: int=10,
	activation: Union[str, NoneType]='softmax'
)
-> tensorflow.python.keras.engine.training.Model
```
Creates a Wide Residual Network with specified parameters.


<h3>Args:</h3>


* **input_shape**: The size of the input tensor (height, width, channels).

* **depth**: Depth of the network. Compute N = (n - 4) / 6. For a depth of 16, n = 16, N = (16 - 4) / 6 = 2 For a depth of 28, n = 28, N = (28 - 4) / 6 = 4 For a depth of 40, n = 40, N = (40 - 4) / 6 = 6

* **widen_factor**: Width of the network.

* **dropout**: Adds dropout if value is greater than 0.0.

* **classes**: The number of outputs the model should generate.

* **activation**: activation function for last dense layer. 

<h3>Raises:</h3>


* **ValueError**: If (depth - 4) is not divisible by 6.

<h3>Returns:</h3>

<ul class="return-block"><li>    A Keras Model.

</li></ul>

