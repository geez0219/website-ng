

### bar_custom
```python
bar_custom(current, total, width=80)
```
Return progress bar string for given values in one of three styles depending on available width:    [..  ] downloaded / total    downloaded / total    [.. ]
* **If total value is unknown or <= 0, show bytes counter using two adaptive styles** :     %s / unknown    %sIf there is not enough space on the screen, do not display anything returned string doesn't include controlcharacters like 