## callback_progress<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/util/wget_util.py/#L110-L141>View source on Github</a>
```python
callback_progress(
	blocks: int,
	block_size: int,
	total_size: int,
	bar_function: Callable[[int, int, int], str]
)
-> None
```
Callback function for urlretrieve that is called when a connection is created and then once for each block.

Draws adaptive progress bar in terminal/console.

Use sys.stdout.write() instead of "print", because it allows one more symbols at the line end without triggering a
linefeed on Windows.

```python
import wget
wget.callback_progress = fe.util.callback_progress
wget.download('http://url.com', '/save/dir', bar=fe.util.bar_custom)
```


<h3>Args:</h3>

* **blocks** :  number of blocks transferred so far.
* **block_size** :  in bytes.
* **total_size** :  in bytes, can be -1 if server doesn't return it.
* **bar_function** :  another callback function to visualize progress.

