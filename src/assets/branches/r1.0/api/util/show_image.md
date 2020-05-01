

### show_image
```python
show_image(im:Union[numpy.ndarray, ~Tensor], axis:matplotlib.axes._axes.Axes=None, fig:matplotlib.figure.Figure=None, title:Union[str, NoneType]=None, color_map:str='inferno') -> Union[matplotlib.figure.Figure, NoneType]
```
Plots a given image onto an axis.

#### Args:

* **axis** :  The matplotlib axis to plot on, or None for a new plot.
* **fig** :  A reference to the figure to plot on, or None if new plot.
* **im** :  The image to display (width X height).
* **title** :  A title for the image.
* **color_map** :  Which colormap to use for greyscale images.