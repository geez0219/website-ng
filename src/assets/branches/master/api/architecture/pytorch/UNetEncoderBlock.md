## UNetEncoderBlock
```python
UNetEncoderBlock(in_channels:int, out_channels:int) -> None
```
A UNet encoder block.

This class is intentionally not @traceable (models and layers are handled by a different process).


#### Args:

* **in_channels** :  How many channels enter the encoder.
* **out_channels** :  How many channels leave the encoder.