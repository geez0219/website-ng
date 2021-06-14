## Rule to write Tutorial and Apphub
we parse the ipynb files in fastestimator/tutorial and fastestimator/apphub to generate markdown for website tutorial
and example section. Because the jupyter notebook already adopts markdown syntax, pretty much everything in ipynb files
works will work in the markdown file. But we still need to watch out superlinks.

* link to url
* link to tutorial/apphub ipynb file
    - with or without anchor tag
* link to current page with anchor tag
* link to API section
* link to assets

### link to url
link to url doesn't need to change at all.

### link to page (tutorial/apphub)
the asset and repo relative file system are the same, so the notebook file support link to other notebook file under
tutorial and apphub folder

### link to current page


