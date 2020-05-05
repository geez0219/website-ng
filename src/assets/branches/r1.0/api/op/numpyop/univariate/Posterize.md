## Posterize
```python
Posterize(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, num_bits:Union[int, Tuple[int, int], Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]=4)
```
Reduce the number of bits for each color channel

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **num_bits** :  Number of high bits. If num_bits is a single value, the range will be [num_bits, num_bits]. A triplet            of ints will be interpreted as [r, g, b], and a triplet of pairs as [[r1, r1], [g1, g2], [b1, b2]]. Must be            in the range [0, 8].
* **Image types** :         uint8    