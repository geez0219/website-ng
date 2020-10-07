## show_image<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L485-L586>View source on Github</a>
```python
show_image(
	im: Union[numpy.ndarray, ~Tensor],
	axis: matplotlib.axes._axes.Axes=None,
	fig: matplotlib.figure.Figure=None,
	title: Union[str, NoneType]=None,
	color_map: str='inferno',
	stack_depth: int=0
)
-> Union[matplotlib.figure.Figure, NoneType]
```
Plots a given image onto an axis.


<h3>Args:</h3>

* **axis** :  The matplotlib axis to plot on, or None for a new plot.
* **fig** :  A reference to the figure to plot on, or None if new plot.
* **im** :  The image to display (width X height).
* **title** :  A title for the image.
* **color_map** :  Which colormap to use for greyscale images.
* **stack_depth** :  Multiple images can be drawn onto the same axis. When stack depth is greater than zero, the `im`        will be alpha blended on top of a given axis.

