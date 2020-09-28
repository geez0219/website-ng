

### show_image
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
Plots a given image onto an axis. The repeated invocation of this function will cause figure plot overlap.

If `im` is 2D and the length of second dimension are 4 or 5, it will be viewed as bounding box data (x0, y0, w, h,
&lt;label&gt;).

```python
boxes = np.array([[0, 0, 10, 20, "apple"],
                  [10, 20, 30, 50, "dog"],
                  [40, 70, 200, 200, "cat"],
                  [0, 0, 0, 0, "not_shown"],
                  [0, 0, -10, -20, "not_shown2"]])

img = np.zeros((150, 150))
fig, axis = plt.subplots(1, 1)
fe.util.show_image(img, fig=fig, axis=axis) # need to plot image first
fe.util.show_image(boxes, fig=fig, axis=axis)
```

Users can also directly plot text

```python
fig, axis = plt.subplots(1, 1)
fe.util.show_image("apple", fig=fig, axis=axis)
```


#### Args:

* **axis** :  The matplotlib axis to plot on, or None for a new plot.
* **fig** :  A reference to the figure to plot on, or None if new plot.
* **im** :  The image (width X height) / bounding box / text to display.
* **title** :  A title for the image.
* **color_map** :  Which colormap to use for greyscale images.
* **stack_depth** :  Multiple images can be drawn onto the same axis. When stack depth is greater than zero, the `im`        will be alpha blended on top of a given axis.

#### Returns:
    plotted figure. It will be the same object as user have provided in the argument.