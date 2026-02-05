
<p align='center'>
  <br/>
  <img src="./docs/logo/organic_logo.png" width="320" height=auto alt=>
  <br/>
</p>

**O**bject **R**econstruction with **G**enerative **A**dversarial **N**etworks from **I**nterferometri**C** data.
You can find the associated paper [here](https://ui.adsabs.harvard.edu/abs/2020SPIE11446E..1UC/abstract). Originally
developed by [Jacques Kluska](https://www.linkedin.com/in/jacques-kluska/) and [Rik Claes](https://www.linkedin.com/in/rik-claes-70a6b71a3/?originalSubdomain=be),
currently developed and maintained by [Toon De Prins](https://deprinst.github.io/). 

# Purpose

The goal of this image reconstruction algorithm is to use an astrophysical Bayesian prior to reconstruct an image from optical (infrared and visible) interferometric data.
Neural networks can learn the main characteristics of the image from the models and guide the image reconstruction to a similar image.
The data itself can induce more complex features if they are indeed needed to reproduce the data.

# Installation
## For CPU

To install ORGANIC in e.g. a Conda environment, you should type the following in a terminal of your choice:

```bash
# create conda enironment
conda create -n organicEnv python=3.10

# activate conda environment
conda activate organicEnv

# install latest ORGANIC version and its dependencies from the PyPI repo
pip install organicoi
```

This will install standard available version of ORGANIC and its dependencies, and should normally
be enough to get ORGANIC working on CPU (though Tensorflow might cause some issues due to the less
than stellar maintenance of its own version dependency lists).

## For GPU
Tensorflow, the neural network package upon which ORGANIC has been built, natively supports
the use of NVIDIA GPU's if the proper version is downloaded and the necessary CUDA libraries
(such as cuDNN) are available on your system. It even provides a specific GPU-version 
including these libraries via `tensorflow[and-cuda]`.

Nevertheless, the Tensorflow GPU version is finicky to install, with different (sub-)versions throwing
different errors at runtime and sometimes not even running at all (including due to version incompatibilities
with Numpy which have not been flagged properly). 

The following install instructions have proven to work using Conda version 24.5.0
and pip version 24.2 on Ubuntu 22.04 (results may differ for your machine/OS):

```bash
# create conda enironment
conda create -n organicEnv python=3.10

# activate conda environment
conda activate organicEnv

# install tensorflow
pip install tensorflow[and-cuda]==2.15.1

# install other dependencies
conda install -c conda-forge scipy matplotlib scikit-learn astropy

# downgrade numpy version
# (the size of numpy.dtype changed at some point, causing an incomatibility
# the tensorflow C/C++ modules)
pip install numpy==1.26.4
```

With this setup Tensorflow will still throw some errors when performing an ORGANIC run on GPU,
though none of these errors are fatal, and often correspond to known bugs due to Tensorflow's error
logging.

# Image reconstruction

To perform an image reconstruction you will have to follow a few steps:

1. Loading the Neural Network
Set the paths and name of the neural network discriminator (`dis`) and generator (`gen`).
There is a pretrained network for disks as seen in the near-infrared (in the `theGANextended2/saved_models` folder).

```python
import organic.organic as org

thegan = org.GAN(dis=dis, gen=gen, adam_lr=0.0001, adam_beta1=0.91, amsgrad=True)
```

- `adam_lr` is the learning rate.
- `adam_beta1` is the exponential decay of the first moment estimates.
- `amsgrad` is whether to apply AMSGrad variant of Adam.
More information on the Adam optimizer can be found [here](https://keras.io/api/optimizers/adam/)

2. Set the SPARCO parameters
SPARCO is an approach allowing to model the star(s) as a geometrical model and image the environment only.
This improves the image of the stellar environment and takes into account the spectral index difference between the stars and the environment.

```python
sparco = org.sparco(fstar=0.61, dstar=-4.0, udstar=0.01, denv=0.0, fsec=0.0,
                        dsec=-4, xsec = 0.0, ysec = 0.0, wave0=1.65e-6,)
```

with:
- `fstar` the stellar-to-total flux ratio at `wave0`
- `dstar` the spectral index of the secondary (if the star is assumed to be Rayleigh-Jeans then lambda^-4^ and `dstar` should be set to -4)
- `udstar` the uniform disk diameter of the primary (in mas)
- `denv` the spectral index of the environment
- `fsec` the secondary-to-total flux ratio
- `dsec` the secondary star spectral index
- `xsec` the ra position of the secondary relative to the primary (in mas)
- `ysec` the dec position of the secondary relative to the primary (in mas)
- `wave0` the reference wavelength at which the flux ratio's are defined (in m)

For more information about SPARCO read [the corresponding paper](https://ui.adsabs.harvard.edu/abs/2014A%26A...564A..80K/abstract) or [this one (application to a circumbinary disk)](https://ui.adsabs.harvard.edu/abs/2016A%26A...588L...1H/abstract).

The SPARCO parameters can be obtained either by making a grid on them with an image reconstruction algorithm like ORGANIC or using geometrical model fitting like [PMOIRED](https://github.com/amerand/PMOIRED).

3. Perform the image reconstruction

```python
thegan.image_reconstruction(datafiles, sparco, data_dir=data_dir, mu=1, ps=0.6, diagnostics=False, epochs=50, nrestar=50, name='output1', boot=False, nboot=100, )
```

with:
- `datafiles` being the name of the file or files, like `*.fits` for example will select all the files ending with.fits in the `data_dir`
- `data_dir` is the path to the data files
- `sparco` is the sparco object defined in point 2.
- `mu` is the hyperparameter giving more or less weight to the Bayesian prior term (usually 1 works well)
- `ps` pixels size in the image (in mas)
- `diagnostics` if True will plot image and convergence criterium for each restart
- `epochs` number of optimizing steps for a givern restart
- `nrestart` number of restarts. starting from a different point in the latent space.
- `name` the name of the output directory that is created
- `boot` if True it will perform a bootstrapping loop where the data will be altered by randomnly drawing new datasets from existant measurements.
- `nboot` number of new datasets to be drawn.

4. Perform a grid on image reconstruction parameters.

ORGANIC is optimized for easy parameter exploration through grids.
It's simple: just make a list of values of any given parameter in `thegan.image_reconstruction` or `sparco`.
This will automatically make image reconstructions corresponding to each parameter combination,
creating folders for each combination of parameters with the value of the parameters reflected
in the name of the folder.

A small example script can be found in `examples/img_rec_example.py` in the ORGANIC project directory.

> [!WARNING]
> Be careful in the interpretation of the images' coordinates. At first sight, it
> might seem that the point `(x, y) = (0, 0)`, which is also the phase center, might 
> correspond to the geometric center of the image. If the image has an even amount of 
> pixels (as is the case in the ORGANIC image reconstructions), this would be on the
> vertex between the central our pixels. This is however not the case in ORGANIC, where 
> due to API reasons the phase center is instead defined half a pixel up and to the
> left of this vertex. This means the FOV of an ORGANIC image is not symmetric w.r.t.
> the point `(0, 0)`, but is instead slightly larger on the negative ends (westward and
> southward). If you do not take this into account you might make errors in the
> plotting of SPARCO components' positions relative to the ORGANIC image.
> Take this discrepancy into account when directly comparing ORGANIC+SPARCO runs
> to image reconstructions of other codes, where `(x, y) = (0, 0)` might instead be
> defined as the geometric center of the image.
> For convenience, we provide the function `img_get_sky_coordinates`, which when
> given an image and pixelscale correctly calculates the sky positions of the image 
> pixel centers (x being positive towards the east/left, y being postive towards the 
> north/top). This can then be used to correctly plot the image, e.g.:

```python
ps = 0.5 # pixelscale 0.5 mas
xcoords, ycoords = org.img_get_sky_coordinates(img, ps=ps)

fig, ax = plt.subplots(1, 1)

# Note we have to add half pixels to each edge to get the extent keyword correct.
# This is because `img_get_sky_coordinates` gives the coordinates of the pixel centers,
# not of their edges, and it is the edges of the outermost pixels that defines the FOV.
ax.imshow(img,
  extent = (
    np.max(xcoords) + ps / 2,
    np.min(xcoords) - ps / 2,
    np.min(ycoords) - ps / 2,
    np.max(ycoords) + ps / 2,
  )
)
plt.show()
```

# Training the neural network (under development)

To train a neural network you need to generate many (~1000) images. Then you can ues them to train the neural network. You need to put them in a fits format as cube of images the third dimenstion being the number of images.
A small example script can be found in `examples/train_gan_example.py` in the ORGANIC project directory.


1. Initialise the GAN

First you need to initialise an empty GAN:

```python
mygan = org.GAN()
```

2. Load your images

The images should be **128 by 128 pixels** (unless you change the shape of the neural network internally...).
ORGANIC will take care of augmenting your data with random rotations, shifts and rescaling.
You can load a cube of images using this command:

```python
imgs = org.InputImages(dir, file, width_shift_range=0, height_shift_range=0,zoom_range=[0.8,1.3],)
```

where:
- `dir` is the directory with the image cube
- `file` is the name of the fits file with the image cube
- `width_shift_range` will shift your image in the horizontal axis by this number of pixels
- `height_shift_range` will shift your image in the vertical axis by this number of pixels
- `zoom_range=[1.8,2.3]` will apply a random zoom factor in this ranges (e.g., smaller than 1 makes your image smaller)

3. Train your GAN

# Pre-trained GANs

- **theGANextended2** MCMax model circumstellar disks with extended component /data/leuven/334/vsc33405/summerjobTests/GANspirals/saved_models
- **GANspirals** geometrical pinwheel nebula models /data/leuven/334/vsc33405/summerjobTests/GANspirals/saved_models

# API documentation
The API documentation (built automatically from the in-source docstrings using [Sphinx](https://www.sphinx-doc.org/en/master/)) is available [here](https://organic.readthedocs.io/en/latest/).

# Use and acknowledgement

If you wish to use ORGANIC in your own research, don't hesitate to contact [Toon De Prins](https://deprinst.github.io/).

If ORGANIC proves useful in your own publications, we kindly ask you to cite the [associated paper](https://ui.adsabs.harvard.edu/abs/2020SPIE11446E..1UC/abstract)
and (optionally) provide a link to the [GitHub repository](https://github.com/DePrinsT/distroi) in the footnotes of your publications.

# Issues and contact

If you face any issues when installing or using ORGANIC, either report them at the project's
[GitHub issues page](https://github.com/DePrinsT/distroi/issues) or please contact [Toon De Prins](https://deprinst.github.io/).

# Developers and contributors

**Developers**

- [Jacques Kluska](https://www.linkedin.com/in/jacques-kluska/) (original developer)
- [Rik Claes](https://www.linkedin.com/in/rik-claes-70a6b71a3/?originalSubdomain=be) (original developer)
- [Toon De Prins](https://deprinst.github.io/) (current lead developer)

**Contributors**

- [Akke Corporaal](https://www.linkedin.com/in/akke-corporaal-19148413b/)

# License

ORGANIC is free software: you can redistribute it and/or modify it under the terms of the MIT license.
ORGANIC is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have received a copy of the MIT license along with ORGANIC.
If not, see <https://opensource.org/license/mit>.
