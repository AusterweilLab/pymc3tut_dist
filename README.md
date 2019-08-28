## PyMC3 Tutorial

Installation instructions:

Readers should install Python 3. Aside from the slight differences in syntax, seeding
is an issue for python 2.7  -- in particular, inconsistent results can occur despite the seeding of rng). 

The easiest way to install PyMC3 would be through the Python Package Index (commonly referred to as "using pip"). This can be performed by simply executing:

```
pip install pymc3
```

By default, the installation process should already include the latest version of other required packages, so no further action should be required. The one exception is to ensure that the `matplotlib` plotting package is installed, since some steps in the provided scripts involve creating plots.  

```
pip install matplotlib
```

Instructions for using the GPU:
On Linux and MacOS systems,  we can specify the desired behavior (e.g., to use the GPU) as a THEANO\_FLAGS environment variable. For instance:
$ THEANO_FLAGS=device=cuda0 python yourscript.py
	  
To run the same script without using the GPU, simply remove the Theano flag:
$ python yourscript.py

On Windows systems, it may be necessary to create a separate Theano configuration file before running the script. To do this, in your home directory, create a
text file named .theanorc (e.g., C:\Users\[your_username]\.theanorc) and include the following in the configuration file:

	[global]
	device = cuda0
	
	[gpuarray]
	preallocate = 1


## Troubleshooting

This is the section to read if you encounter any issues while trying to follow the instructions in the tutorial. Beyond that, the recommended action would be to explore the pymc3 discourse page (https://discourse.pymc.io/about).


In general, a large number of errors can be resolved by making sure that each of the key Python packages (e.g., pymc3, theano, numpy) are kept up to date. Occasionally (especially with support for Theano being reduced to a minimum), it may be more effective to keep certain packages running at a lower version. 


Unfortunately, error messages from Python package conflicts can be cryptic. For example, one package conflict that we have encountered is

```
AttributeError: module 'numpy.core.multiarray' has no attribute '_get_ndarray_c_version'
```

One of the authors (JLA) encountered this error when their Theano version was 1.0.3 and numpy was 1.15. It was resolved by upgrading Theano to 1.0.4 and numpy to 1.16.

### Nesting bracket error
When sampling from large hierarchical models, you may encounter a nesting bracket error. One way to address this is to simply add this line after importing theano:

```
theano.config.gcc.cxxflags = "-fbracket-depth=1024"
```