## UNetDecoderBlock
```python
UNetDecoderBlock(in_channels:int, mid_channels:int, out_channels:int) -> None
```
A UNet decoder block.

This class is intentionally not @traceable (models and layers are handled by a different process).


#### Args:

* **in_channels** :  How many channels enter the decoder.
* **mid_channels** :  How many channels are used for the decoder's intermediate layer.
* **out_channels** :  How many channels leave the decoder.