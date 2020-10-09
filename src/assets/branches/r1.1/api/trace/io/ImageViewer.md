## ImageViewer<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/io/image_viewer.py/#L27-L61>View source on Github</a>
```python
ImageViewer(
	inputs: Union[str, Sequence[str]],
	mode: Union[str, Set[str]]=('eval', 'test'),
	width: int=12,
	height: int=6
)
-> None
```
A trace that interrupts your training in order to display images on the screen.

This class is useful primarily for Jupyter Notebook, or for debugging purposes.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be displayed.

* **mode**: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **width**: The width in inches of the figure.

* **height**: The height in inches of the figure.

