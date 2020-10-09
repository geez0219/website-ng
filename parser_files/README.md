## How to write docstring

### Scenario 1: url is too long to fit in 120 limit
If url is too long and need to be separated to muliple lines, users need to wrap that url with ''. NOTE: not all string works, only url will work.

\<url\> => '\<url\>'

example:
```
"""
    See 'https://papers.nips.cc/paper/9070-error-correcting-output-codes-improve-probability-estimation-and-adversarial-
    robustness-of-deep-neural-networks'
"""
```

### Scenario 2: use string to represent flow chart
Because everything in the docstring will get converted to markdown, multiple space will no long mean the same number of spaces. If users want to use ASCII to draw flow chart. They need to wrap that content like this:

\'\'\' plot

\<content\>

\'\'\'

example:
```
"""
``` plot
            Training:                                       Testing:

        on_begin                                            on_begin
            |                                                   |
        on_epoch_begin (train)  <------<                    on_epoch_begin (test)  <------<
            |                          |                        |                         |
        on_batch_begin (train) <----<  |                    on_batch_begin (test) <----<  |
            |                       |  |                        |                      |  |
        on_batch_end (train) >-----^   |                    on_batch_end (test) >------^  |
            |                          ^                        |                         |
        on_epoch_end (train)           |                    on_epoch_end (test) >---------^
            |                          |                        |
        on_epoch_begin (eval)          |                    on_end
            |                          ^
        on_batch_begin (eval) <----<   |
            |                      |   |
        on_batch_end (eval) >-----^    |
            |                          |
        on_epoch_end (eval) >----------^
            |
        on_end
\```(without backslash)
"""
```

### Scenario 3: need to represent python magic_methods
A magic methods has four underscores around a text (two for each side) and this is also the pattern for markdown to make the text bold. Users can wrap the magic method with ``.

example:

``` python
"""
    `__init__`
"""
```