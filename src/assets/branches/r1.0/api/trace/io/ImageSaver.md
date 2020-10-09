## ImageSaver<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/io/image_saver.py/#L26-L65>View source on Github</a>
```python
ImageSaver(
	inputs: Union[str, Sequence[str]],
	save_dir: str='/home/geez219/angular_project/website-ng2/parser_files',
	dpi: int=300,
	mode: Union[str, Set[str]]=('eval', 'test')
)
-> None
```
A trace that saves images to the disk.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be saved.

* **save_dir**: The directory into which to write the images.

* **dpi**: How many dots per inch to save.

* **mode**: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

