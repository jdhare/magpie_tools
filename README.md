# magpie_tools
Python based tools for MAGPIE data analysis. There is the stable branch with code that is known to work, development will take place on another branch.

There are three folders in this repo:

* Code: this contains code and class definitions to support the notebooks in the other two folders. It can be used in other notebooks, or other python programs.
* Tools: These can be used in their original folder, for simple tasks that don't require a paper trail.
* Templates: You should copy these to another folder outside of the repository before using them. Each template should be customised for a specific shot you are analysing. You should always copy from the repository folder to ensure you get the latest version.

If you modify any of the tools or templates, be aware that these changes will be overwritten by any updates. So do not store any data here
you are not prepared to lose - instead, make a copy of the file you want to change. Or you could fork the repo, make changes and submit a pull request if you think your changes will benefit everyone!

In order to run these jupyter notebooks, use shift+right click>"open command window here" in the folder containing the notebook you want to use. Type 'jupyter notebook' at the command line and hit enter.

### Current code:
* **shot_statistics**: Can grab scope data from shots between a start and end date, and get statistics on trigger Marx timings, line gap switch spreads and integrated MITL b-dot signals. Useful for checking whether you got a 'good' current on your last shot, or to compare with Rogowski measurements. Quite buggy as occasionally there is no data from a scope and I haven't handled the errors well.
* **plasma_parameter_calculator**: Calculates all sorts of useful parameters, from the Sonic Mach number to the viscosity, as well as mean free paths using the more complicated formulas from the NRL formulary.

### Current tools:
* **fringe_tracing_fourier**: semi-automated fringe tracing based on the 2D Fourier transform. Will get the background interferogram perfectly, and will get unperturbed fringes in the shot interferogram pretty well. Saves a parameter log file for future reference.
* **fringe_tracing_fourier_circular**: semi-automated fringe tracing for circular fringe patterns, similarly based on the 2D Fourier transform. This won't necessarily trace the interferogram perfectly (even for the background), but can reduce the amount of hand tracing required. Saves a parameter log file for future reference.
* **fringe_tracing_smoothing**: a new, automated fringe tracing method based on smoothing the fringes' contours to make thinning work better. Highlights possible errors to make corrections by hand easier. Should get most of the interferogram well.
* **shiftr**: used to produced a set of 12 frame images that have been shifted so they all overlap

### Current templates:
* **faraday_template**: takes you from raw images to a polarogram, then overlays the interferogram onto that, and then you can add in a processed electron density map to get the magnetic field map. Lots of image registration!
* **fast_frame**: takes images from shiftr and allows you to look at lineouts, play with levels and save out animated gifs.
* **thomson**: fitting for Thomson scattering spectra in .asc format, exported from the Andor spectrometer. Uses an nLTE model to decompose Z and T_e, and offers full flexibility over specifying independent and dependent variables in fits.
* **background_reconstruction**: reconstructs a full background phase map by fitting a Gaussian optics model to empty regions of the interpolated shot phase map. Useful when the background fringes are rotated compared to the empty regions in the shot, which produces a gradient across the fringe shift map.


### What you need:
Anaconda is probably the best distribution to use: https://www.anaconda.com/download/ - get the 64 bit version with the latest python kernel (3.6 as of writing). You may find that some packages you need are missing, but these can be installed using `conda install X` at the command line - ask for help if you need it. You will need at least:

`conda install --channel https://conda.anaconda.org/menpo opencv3`

`conda install -c conda-forge imageio`

`conda install -c conda-forge lmfit`

The image registration algorithms are installed differently. At the command line, run:

`pip install git+https://github.com/matejak/imreg_dft.git`

In order for Python to find the code in this respository, you must add the folder where you put the code to the PATH:
* Press the windows key and type 'environment'. Click 'Edit the system envrionment variables'
* Click 'Environment Variables' in the window which appears.
* Under 'System Variables' locate PYTHONPATH. Double click it.
* Click 'New'. Paste the location of the folder containing this code, eg. C:\Users\jdhare\Documents\GitHub\magpie_tools
* Click okay on the three windows which opened since you started this process.
* Restart your jupyter notebook server (it'll be a command prompt window titled 'jupyter notebook'.
