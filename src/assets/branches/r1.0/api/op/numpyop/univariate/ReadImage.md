## ReadImage<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/numpyop/univariate/read_image.py/#L24-L72>View source on Github</a>
```python
ReadImage(
	inputs: Union[str, Iterable[str], Callable],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	parent_path: str='',
	color_flag: Union[str, int]=1
)
```
A class for reading png or jpg images from disk.


<h3>Args:</h3>


* **inputs**: Key(s) of paths to images to be loaded.

* **outputs**: Key(s) of images to be output.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **parent_path**: Parent path that will be prepended to a given path.

* **color_flag**: Whether to read the image as 'color', 'grey', or one of the cv2.IMREAD flags. 

<h3>Raises:</h3>


* **AssertionError**: If `inputs` and `outputs` have mismatched lengths, or the `color_flag` is unacceptable.

