

### callback_progress
```python
callback_progress(blocks:int, block_size:int, total_size:int, bar_function:Callable[[int, int, int], str]) -> None
```
Callback function for urlretrieve that is called when a connection is created and then once for each block.Draws adaptive progress bar in terminal/console.Use sys.stdout.write() instead of "print", because it allows one more symbols at the line end without triggering alinefeed on Windows.```pythonimport wgetwget.callback_progress = fe.util.callback_progress
* **wget.download('http** : //url.com', '/save/dir', bar=fe.util.bar_custom)```

#### Args:

* **blocks** :  number of blocks transferred so far.
* **block_size** :  in bytes.
* **total_size** :  in bytes, can be -1 if server doesn't return it.
* **bar_function** :  another callback function to visualize progress.