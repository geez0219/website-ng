## WideResidualNetwork<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/architecture/pytorch/wideresnet.py/#L95-L150>View source on Github</a>
```python
WideResidualNetwork(
	depth: int=28,
	classes: int=10,
	widen_factor: int=10,
	dropout: float=0.0
)
-> None
```
Wide Residual Network.

This class creates the Wide Residual Network with specified parameters.


<h3>Args:</h3>


* **depth**: Depth of the network. Compute N = (n - 4) / 6. For a depth of 16, n = 16, N = (16 - 4) / 6 = 2 For a depth of 28, n = 28, N = (28 - 4) / 6 = 4 For a depth of 40, n = 40, N = (40 - 4) / 6 = 6

* **classes**: The number of outputs the model should generate.

* **widen_factor**: Width of the network.

* **dropout**: Adds dropout if value is greater than 0.0. 

<h3>Raises:</h3>


* **AssertionError**: If (depth - 4) is not divisible by 6.

