

### parse_modes
```python
parse_modes(modes:Set[str]) -> Set[str]
```
A function to determine which modes to run on based on a set of modes potentially containing blacklist values.```pythonm = fe.util.parse_modes({"train"})  # {"train"}m = fe.util.parse_modes({"!train"})  # {"eval", "test", "infer"}m = fe.util.parse_modes({"train", "eval"})  # {"train", "eval"}m = fe.util.parse_modes({"!train", "!infer"})  # {"eval", "test"}```

#### Args:

* **modes** :  The desired modes to run on (possibly containing blacklisted modes).

#### Returns:
    The modes to run on (converted to a whitelist).

#### Raises:

* **AssertionError** :  If invalid modes are detected, or if blacklisted modes and whitelisted modes are mixed.