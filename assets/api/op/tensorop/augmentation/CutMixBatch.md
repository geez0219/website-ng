## CutMixBatch
```python
CutMixBatch(inputs=None, outputs=None, mode=None, alpha=1.0)
```
This class should be used in conjunction with MixUpLoss to perform CutMix training, which helps to reduceover-fitting, perform object detection, and against adversarial attacks (https://arxiv.org/pdf/1905.04899.pdf)

#### Args:

* **inputs** :  key of the input to be cut-mixed
* **outputs** :  key to store the cut-mixed input
* **mode** :  what mode to execute in. Probably 'train'
* **alpha** :  the alpha value defining the beta distribution to be drawn from during training

### forward
```python
forward(self, data, state)
```
 Forward method to perform cutmix batch augmentation

#### Args:

* **data** :  Batch data to be augmented (batch X height X width X channel)
* **state** :  Information about the current execution context.

#### Returns:
            Cut-Mixed batch data        