

### parse_cli_to_dictionary
```python
parse_cli_to_dictionary(input_list:List[str]) -> Dict[str, Any]
```
Convert a list of strings into a dictionary with python objects as values.```pythona = parse_cli_to_dictionary(["--epochs", "5", "--test", "this", "--lr", "0.74"]) 
* **# {'epochs'** :  5, 'test' 'this', 'lr' 0.74}```

#### Args:

* **input_list** :  A list of input strings from the cli.

#### Returns:
    A dictionary constructed from the `input_list`, with values converted to python objects where applicable.