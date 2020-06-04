

### enable_deterministic
```python
enable_deterministic(seed)
```
Invoke to set random seed for deterministic training. The determinism only works for tensorflow >= 2.1 and
pytorch >= 1.14, and some model layers don't support.

Known failing layers:
* tf.keras.layers.UpSampling2D