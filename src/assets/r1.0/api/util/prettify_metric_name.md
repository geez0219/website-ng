

### prettify_metric_name
```python
prettify_metric_name(metric:str) -> str
```
Add spaces to camel case words, then swap _ for space, and capitalize each word.```pythonx = fe.util.prettify_metric_name("myUgly_loss")  # "My Ugly Loss"```

#### Args:

* **metric** :  A string to be formatted.

#### Returns:
    The formatted version of 'metric'.