

### bar_custom
```python
bar_custom(current:float, total:float, width:int=80) -> str
```
Return progress bar string for given values in one of three styles depending on available width.
* **This function was modified from wget source code at https** : //bitbucket.org/techtonik/python-wget/src/default/.
* **The bar will be one of the following formats depending on available width** :     [..  ] downloaded / total    downloaded / total    [.. ]
* **If total width is unknown or <= 0, the bar will show a bytes counter using two adaptive styles** :     %s / unknown    %sIf there is not enough space on the screen, do not display anything. The returned string doesn't include controlcharacters like  used to place cursor at the beginning of the line to erase previous content.This function leaves one free character at the end of the string to avoid automatic linefeed on Windows.```python
* **wget.download('http** : //url.com', '/save/dir', bar=fe.util.bar_custom)```

#### Args:

* **current** :  The current amount of progress.
* **total** :  The total amount of progress required by the task.
* **width** :  The available width.

#### Returns:
    A formatted string to display the current progress.