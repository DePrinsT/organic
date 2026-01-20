"""Module for loading in and manipulating OI data in OIFITS format."""

import equinox as eqx


# NOTE: should be treated as a fully frozen dataclass in the calculations, i.e.
# no gradients should ever be calculated with respect to it. Cannot use
# static fields with eqx.field(static=True), since the data will have to be JAX
# arrays, which are themselves not hashable.
class OIDataJaxed(eqx.Module):
    """ """

    pass
