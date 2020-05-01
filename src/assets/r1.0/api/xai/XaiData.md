## XaiData
```python
XaiData(**kwargs:~Tensor) -> None
```
A container for xai related data.    This class is useful for automatically laying out collections of images for comparison and visualization.    ```python    d = fe.xai.XaiData(y=tf.ones((4,)), x=0.5*tf.ones((4, 32, 32, 3)))    fig = d.paint_figure()    plt.show()    ```

#### Args:

 **kwargs :  image_title / image pairs for visualization. Images with the same batch dimensions will be laid out            side-by-side, with earlier kwargs entries displayed further to the left.    

### paint_figure
```python
paint_figure(self, height_gap:int=100, min_height:int=200, width_gap:int=50, min_width:int=200, dpi:int=96) -> matplotlib.figure.Figure
```
Visualize the current XaiData entries in a matplotlib figure.        ```python        d = fe.xai.XaiData(y=tf.ones((4,)), x=0.5*tf.ones((4, 32, 32, 3)))        fig = d.paint_figure()        plt.show()        ```

#### Args:

* **height_gap** :  How much space to put between each row.
* **min_height** :  The minimum height of a row.
* **width_gap** :  How much space to put between each column.
* **min_width** :  The minimum width of a column.
* **dpi** :  The resolution of the image to display.

#### Returns:
            The handle to the generated matplotlib figure.        

### paint_numpy
```python
paint_numpy(self, height_gap:int=100, min_height:int=200, width_gap:int=50, min_width:int=200, dpi:int=96) -> numpy.ndarray
```
Visualize the current XaiData entries into an image stored in a numpy array.        ```python        d = fe.xai.XaiData(y=tf.ones((4,)), x=0.5*tf.ones((4, 32, 32, 3)))        img = d.paint_numpy()        plt.imshow(img[0])        plt.show()        ```

#### Args:

* **height_gap** :  How much space to put between each row.
* **min_height** :  The minimum height of a row.
* **width_gap** :  How much space to put between each column.
* **min_width** :  The minimum width of a column.
* **dpi** :  The resolution of the image to display.

#### Returns:
            A numpy array with dimensions (1, height, width, 3) containing an image representation of this XaiData.        