"""Module for loading in and manipulating OI data in OIFITS format."""

import equinox as eqx

# TODO: should maintain some of the internal file-structure and metadata of the OIFITS
# files on the host (i.e. everything should be stored as numpy arrays). We should
# have the ability to go from this internal representation back to a numpy file.
# Before interacting with the other ORGANIC internals, the data should be served
# as a set of 1D JAX arrays (i.e. the data should be commited to the JAX device),
# and be available for JAX transformations. You should be able to specify which data
# you want to receive (i.e. all the VIS values, T3PHI, etc.), and there should
# be support to read these in an to maintain CHI2/Loss values for each.
#
# This should include filtering operations for example in the flags of the data, NaNs or
# in wavelength. Then there should be a way of writing the resulting observables back
# into the OIFITS file-format for further analysis/use elsewhere. This could be
# accomplished by maintaining 1D index arrays linking the observed data-point to
# a specific OIFITS file, baseline/telescope and wavelength/array solution, which can
# probably also an array containing indices of which wavelength points have been
# filtered out, then to be be mapped back onto numpy arrays.


# NOTE: should be treated as a fully frozen dataclass in the calculations, i.e.
# no gradients should ever be calculated with respect to it. Cannot use
# static fields with eqx.field(static=True), since the data will have to be JAX
# arrays, which are themselves not hashable.
class OIDataJaxed(eqx.Module):
    """ """

    pass
