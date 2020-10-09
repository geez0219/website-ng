# tf.cli.cli_util<span class="tag">module</span>

---

## SaveAction<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/cli/cli_util.py/#L8-L43>View source on Github</a>
```python
SaveAction(
	option_strings: Sequence[str],
	dest: str,
	nargs: Union[int, str, NoneType]='?', **kwargs: Dict[str, Any]
)
-> None
```
A customized save action for use with argparse.

A custom save action which is used to populate a secondary variable inside of an exclusive group. Used if this file
is invoked directly during argument parsing.


<h3>Args:</h3>


* **option_strings**: A list of command-line option strings which should be associated with this action.

* **dest**: The name of the attribute to hold the created object(s).

* **nargs**: The number of command line arguments to be consumed.

* ****kwargs**: Pass-through keyword arguments.

---

## parse_cli_to_dictionary<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/cli/cli_util.py/#L46-L77>View source on Github</a>
```python
parse_cli_to_dictionary(
	input_list: List[str]
)
-> Dict[str, Any]
```
Convert a list of strings into a dictionary with python objects as values.

```python
a = parse_cli_to_dictionary(["--epochs", "5", "--test", "this", "--lr", "0.74"]) 
# {'epochs': 5, 'test': 'this', 'lr': 0.74}
```


<h3>Args:</h3>


* **input_list**: A list of input strings from the cli. 

<h3>Returns:</h3>

<ul class="return-block"><li>    A dictionary constructed from the <code>input_list</code>, with values converted to python objects where applicable.</li></ul>

