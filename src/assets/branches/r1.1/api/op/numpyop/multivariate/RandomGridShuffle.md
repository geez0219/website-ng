## RandomGridShuffle<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/multivariate/random_grid_shuffle.py/#L24-L64>View source on Github</a>
```python
RandomGridShuffle(
	grid: Tuple[int, int]=(3, 3),
	mode: Union[str, NoneType]=None,
	image_in: Union[str, NoneType]=None,
	mask_in: Union[str, NoneType]=None,
	masks_in: Union[str, NoneType]=None,
	image_out: Union[str, NoneType]=None,
	mask_out: Union[str, NoneType]=None,
	masks_out: Union[str, NoneType]=None
)
```
Divide an image into a grid and randomly shuffle the grid's cells.


<h3>Args:</h3>

* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **image_in** :  The key of an image to be modified.
* **mask_in** :  The key of a mask to be modified (with the same random factors as the image).
* **masks_in** :  The key of masks to be modified (with the same random factors as the image).
* **image_out** :  The key to write the modified image (defaults to `image_in` if None).
* **mask_out** :  The key to write the modified mask (defaults to `mask_in` if None).
* **masks_out** :  The key to write the modified masks (defaults to `masks_in` if None).
* **grid** :  size of grid for splitting image (height, width).
* **Image types** :     uint8, float32



