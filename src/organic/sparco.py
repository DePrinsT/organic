import abc

import equinox as eqx
import jax


class GeometricComponent(eqx.Module):
    """A fully abstract class representing geometric components (e.g. for use in
    SPARCO).

    The parameters defining the geometric component are to be stored in the relevant
    instance attributes. These can be either single-element JAX arrays or just Python
    floats. In the latter case, Organic will consider them fixed during any
    optimization."""

    @abc.abstractmethod
    def get_complex_visibilities(self, u: jax.Array, v: jax.Array) -> jax.Array:
        r"""Compute complex visibilities.

        **Arguments**

        - `u`: 1D array of $u$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.
        - `v`: 1D array of $v$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.

        **Returns**

        A complex 1D array holding the complex visibilities calculated at the specified
        $uv$-coordinates.
        """
        raise NotImplementedError


class Sparco(eqx.Module):
    """
    Docstring for Sparco

    **Attributes**

    - `components`: A tuple of geometric model components.
    """

    components: tuple[GeometricComponent, ...]


class UniformDisk(GeometricComponent):
    r"""A uniform dis.

    **Attributes**

    - `x`: The $x$-position of the source in $\mathrm{mas}$.
    - `y`: The $y$-position of the source in $\mathrm{mas}$.
    - `ud`:  The angular uniform disk size in $\mathrm{mas}$.
    - `sp_idx`: The spectral index $F_lambda \propto \lamba^\mathrm{sp_idx}$
        of the source.

    The attributes can be initialised as either JAX arrays, in which case they are free
    to be optimised, or floats, in which they are considered static."""

    # Required instance attributes.
    x: jax.Array | float
    y: jax.Array | float
    ud: jax.Array | float
    sp_idx: jax.Array | float

    def __init__(
        self,
        *,
        x: jax.Array | float = 0.0,
        y: jax.Array | float = 0.0,
        ud: jax.Array | float,
        sp_idx: jax.Array | float = 0.0,
    ) -> None:
        """Mapped directly to class attributes, please see the class docstring."""
        self.x = x
        self.y = y
        self.ud = ud
        self.sp_idx = sp_idx

    def get_complex_visibilities(self, u: jax.Array, v: jax.Array) -> jax.Array:
        r"""Compute complex visibilities.

        **Arguments**

        - `u`: 1D array of $u$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.
        - `v`: 1D array of $v$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.

        **Returns**

        A complex 1D array holding the complex visibilities calculated at the specified
        $uv$-coordinates.
        """
        # TODO: implement
        raise NotImplementedError
