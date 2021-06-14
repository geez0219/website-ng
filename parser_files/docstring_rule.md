## Why we need docstring
Every website page under API tag is created by parsing the docstring of the FE repo. The parser (`fe_parser.py`) will
parse every python file under fastestimator/fastestimator folder and generate the markdown file. The website will render
the markdown file dynamically.

## Where need docstring
Ideally all functions and classes need docstrings, but private ones (name start with "_") don't neccesarily need it.
Only the functions and classes with names that not start with "_" will be parsed to the website API section.

## How to write docstring
Google docstring style (https://google.github.io/styleguide/pyguide.html) is the one we must followed due to the way
`fe_parser.py` designed.

basic structure:
```
    """<main_description>

    Args:
        <arg1_variable>: <arg1_description>
        <arg2_variable>: <arg2_description>
        ...

    Returns:
        <returns_description>

    Raises:
        <error1_type>: <error1_description>
        <error2_type>: <error1_description>
    """
```

Note:
* All items under `Args`, `Returns`, `Raises` need to have exact 4 preceeding space and the variable name need to be
    trailed with ": "
* <main_description> can start at the first line without leading space.
* First character of <item_description> need to be capitalized
* The parser will parse the docstring to markdown, so some markdown sytax can be used (ex: emphasis)

### Scenario 1: descriptions are too long to fit in a single line
If some descriptions are too long and need to break the lines, writer need to break the line with different way depending
on the context.

#### break the line in <main_description>
Directly break the line (don't need trailing "\"). The next line starting character need to allign to the first " in """

example:
```
"""this is the first line
this is the next line
"""
```

#### break the line in <item_description>
Directly break the line (don't need trailing "\") and put 4 spaces before the new line. The new line starting character
needs to allign to the position where the previous line starts plus 4 spaces.

example:
```
"""
    Args: this is the first line
        this is the next line
"""
```

#### break url to multiple line
If url is too long and needs to be separated to muliple lines, users need to wrap that url with ''. NOTE: not all string
works, only url will work.

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