## Cropping2D
```python
Cropping2D(cropping:Union[int, Tuple[Union[int, Tuple[int, int]], Union[int, Tuple[int, int]]]]=0) -> None
```
A layer for cropping along height and width dimensions    ```python    x = torch.tensor(list(range(100))).view((1,1,10,10))    m = fe.layers.pytorch.Cropping2D(3)    y = m.forward(x)  # [[[[33, 34, 35, 36], [43, 44, 45, 46], [53, 54, 55, 56], [63, 64, 65, 66]]]]    m = fe.layers.pytorch.Cropping2D((3, 4))    y = m.forward(x)  # [[[[34, 35], [44, 45], [54, 55], [64, 65]]]]    m = fe.layers.pytorch.Cropping2D(((1, 4), 4))    y = m.forward(x)  # [[[[14, 15], [24, 25], [34, 35], [44, 45], [54, 55]]]]    ```

#### Args:

* **cropping** :  Height and width cropping parameters. If a single int 'n' is specified, then the width and height of            the input will both be reduced by '2n', with 'n' coming off of each side of the input. If a tuple ('h', 'w')            is provided, then the height and width of the input will be reduced by '2h' and '2w' respectively, with 'h'            and 'w' coming off of each side of the input. If a tuple like (('h1', 'h2'), ('w1', 'w2')) is provided, then            'h1' will be removed from the top, 'h2' from the bottom, 'w1' from the left, and 'w2' from the right            (assuming the top left corner as the 0,0 origin).

#### Raises:

* **ValueError** :  If `cropping` has an unacceptable data type.    