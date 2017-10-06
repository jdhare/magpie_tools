# magpie_tools
Python based tools for MAGPIE data analysis. There is the stable branch with code that is known to work, development will take place on another branch.

There are three folders in this repo:

* Code: this contains code and class definitions to support the notebooks in the other two folders. It can be used in other notebooks, or other python programs.
* Tools: These can be used in their original folder, for simple tasks that don't require a paper trail. 
* Templates: You should copy these to another folder outside of the repository before using them. Each template should be customised for a specific shot you are analysing. You should always copy from the repository folder to ensure you get the latest version.

If you modify any of the tools or templates, be aware that these changes will be overwritten by any updates. So do not store any data here
you are not prepared to lose - instead, make a copy of the file you want to change. Or you could fork the repo, make changes and submit a pull request if you think your changes will benefit everyone!

In order to run these jupyter notebooks, use shift+right click>"open command window here" in the folder containing the notebook you want to use. Type 'jupyter notebook' at the command line and hit enter.

### Current tools:
* **fringe_tracing**: semi-automated fringe tracing. Will get the background interferogram perfectly, and will get unperturbed fringes in the shot interferogram pretty well.
* **shiftr**: used to produced a set of 12 frame images that have been shifted so they all overlap

### Current templates:
* **faraday_template**: takes you from raw images to a polarogram, then overlays the interferogram onto that, and then you can add in a processed electron density map to get the magnetic field map. Lots of image registration!
* **fast_frame**: takes images from shiftr and allows you to look at lineouts, play with levels and save out animated gifs.

### Coming soon(-ish)!
* **thomson**: this requires a complete rewrite, but it was the first python program I really wrote and it's still very powerful
* **xuv**: for aligning images from the 4-frame cameras.
