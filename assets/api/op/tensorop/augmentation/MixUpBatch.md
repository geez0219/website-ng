## MixUpBatch
```python
MixUpBatch(inputs=None, outputs=None, mode=None, alpha=1.0)
```
This class should be used in conjunction with MixUpLoss to perform mix-up training, which helps to reduceover-fitting, stabilize GAN training, and against adversarial attacks (https://arxiv.org/abs/1710.09412)

#### Args:

* **inputs** :  key of the input to be mixed up
* **outputs** :  key to store the mixed-up input
* **mode** :  what mode to execute in. Probably 'train'
* **alpha** :  the alpha value defining the beta distribution to be drawn from during training

### forward
```python
forward(self, data, state)
```
 Forward method to perform mixup batch augmentation

#### Args:

* **data** :  Batch data to be augmented
* **state** :  Information about the current execution context.

#### Returns:
            Mixed-up batch data        