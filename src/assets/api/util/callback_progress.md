

### callback_progress
```python
callback_progress(blocks, block_size, total_size, bar_function)
```
Callback function for urlretrieve that is called when connection is created and when once for each block.Draws adaptive progress bar in terminal/console.Use sys.stdout.write() instead of "print,", because it allows one more symbol at the line end without linefeed on Windows

#### Args:

* **blocks** :  number of blocks transferred so far
* **block_size** :  block size in bytes
* **total_size** :  total size in bytes, can be -1 if server doesn't return it
* **bar_function** :  another callback function to visualize progress