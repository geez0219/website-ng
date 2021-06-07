## prettify_metric_name<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/util/util.py/#L341-L354>View source on Github</a>
```python
prettify_metric_name(
	metric: str
)
-> str
```
Add spaces to camel case words, then swap _ for space, and capitalize each word.

```python
x = fe.util.prettify_metric_name("myUgly_loss")  # "My Ugly Loss"
```


<h3>Args:</h3>


* **metric**: A string to be formatted. 

<h3>Returns:</h3>

<ul class="return-block"><li>    The formatted version of 'metric'.</li></ul>

