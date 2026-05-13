"""Module for defining semi-parametric models following the Semi-Parametric Approach
for the Reconstruction of Chromatic Objects (SPARCO) to be used for imaging of
faint structures surrounding birght central sources."""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from organic._consts import MAS2RAD

# TODO: should also contain the flux value and spectral behaviour of the image
# under reconstruction itself.


class SpectralShape(eqx.Module):
    r"""A fully abstract class representing the shape (e.g. power-law,
    blackbody) of a spectrum $F_{\lambda}$. I.e. a relative spectrum or specific
    intensity profile, not an absolute one.

    The parameters defining the geometric component are to be stored in the relevant
    instance attributes. These can be either single-element JAX arrays or just Python
    floats. In the latter case, Organic will consider them fixed during any
    optimization."""

    @abc.abstractmethod
    def get_flux(
        self, wavelengths: jax.Array, *, wave0: float, f0: float | jax.Array
    ) -> jax.Array:
        r"""Retrieve the $F_{\lambda}$ flux at desired wavelengths given a reference
        wavelength and reference flux.

        **Arguments**

        - `wavelengths`: JAX array containing the wavelengths in $\mathrm{m}$ at which
            to calculate the spectral flux.
        - `wave0`: The reference wavelength in $\mathrm{m}$. We define
            the reference flux `f0` at this point, with the returned flux being
            calculated relative to these reference values.
        - `f0`: The reference $F_{\lambda}$ flux level.

        **Returns**

        A 1D array containing the output $F_{\lambda}$ spectrum according to the
        spectral shape. This is calculated assuming a reference $F_{\lambda}$ flux `f0`
        at a reference wavelength `wave0`. The units of this will be whatever the
        original units of `f0` are (which can also represent flux fractions).
        """
        raise NotImplementedError


class GeometricComponent(eqx.Module):
    r"""A fully abstract class representing geometric components (e.g. for use in
    SPARCO). Concrete instantiations must include a spectral shape instance parameter
    and a method for retrieving complex visibilities at specified spatial frequencies.

    **Attributes**

    - `x`: The $x$-position of the source in $\mathrm{mas}$.
    - `y`: The $y$-position of the source in $\mathrm{mas}$.
    - `spec_shape`: The relative spectral shape of the geometric component.

    The parameters defining the geometric component are to be stored in the relevant
    instance attributes. These can be either single-element JAX arrays or just Python
    floats. In the latter case, Organic will consider them fixed during any
    optimization."""

    # Required abstract attributes.
    x: eqx.AbstractVar[jax.Array | float]  # Component position
    y: eqx.AbstractVar[jax.Array | float]
    spec_shape: eqx.AbstractVar[SpectralShape]  # Abstract spectral shape variable

    @abc.abstractmethod
    def get_complex_visibilities(self, u: jax.Array, v: jax.Array) -> jax.Array:
        r"""Compute complex visibilities (including component's position).

        **Arguments**

        - `u`: 1D array of $u$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.
        - `v`: 1D array of $v$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.

        **Returns**

        A complex 1D array holding the complex visibilities calculated at the specified
        $uv$-coordinates.
        """
        raise NotImplementedError

    def _apply_positional_phase_offset(
        self, vis: jax.Array, u: jax.Array, v: jax.Array
    ) -> jax.Array:
        r"""Apply a phase offset to given array of complex visibilities according
        to the component's position.

        **Arguments**

        - `vis`: 1D array containing complex visibilities calculated for a geometric
            component at position $(x, y) = (0, 0)$.
        - `u`: 1D array of $u$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.
        - `v`: 1D array of $v$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.

        **Returns**

        A complex 1D array holding the complex visibilities calculated at the specified
        $uv$-coordinates.
        """
        # Set component position in radian
        x_rad, y_rad = self.x * MAS2RAD, self.y * MAS2RAD
        return vis * jnp.exp(-2j * jnp.pi * (x_rad * u + y_rad * v))


class Sparco(eqx.Module):
    r"""
    Class representing the geometric components of a SPARCO imaging model.

    **Attributes**

    - `components`: A tuple of geometric model components.
    - `fluxes`: A tuple of fluxes for the model components. When fitting visibilities,
        these are only considered as relative flux fractions between 0 and 1. Otherwise,
        when fitting correlated fluxes, these are considered total fluxes in Jansky.
        These can be either single-element JAX arrays or just Python floats. In the
        latter case, Organic will consider them fixed during any optimization.
    - `wave0`: The central wavelength of the Sparco model components in $\mathrm{m}$.
    """

    # Required instance attributes.
    components: tuple[GeometricComponent, ...]
    fluxes: tuple[float | jax.Array, ...]
    wave0: float

    def __init__(
        self,
        components: tuple[GeometricComponent, ...],
        fluxes: tuple[float | jax.Array, ...],
        *,
        wave0: float,
    ) -> None:
        self.components = components
        self.fluxes = fluxes
        self.wave0 = wave0


class UniformDisk(GeometricComponent):
    r"""A uniform disk.

    **Attributes**

    - `x`: The $x$-position of the source in $\mathrm{mas}$.
    - `y`: The $y$-position of the source in $\mathrm{mas}$.
    - `ud`:  The angular uniform disk size in $\mathrm{mas}$.
    - `spec_shape`: The relative spectral shape of the geometric component.

    The attributes can be initialised as either JAX arrays, in which case they are free
    to be optimised, or floats, in which case they are considered static."""

    # Required instance attributes.
    x: jax.Array | float
    y: jax.Array | float
    ud: jax.Array | float
    spec_shape: SpectralShape

    def __init__(
        self,
        *,
        x: jax.Array | float = 0.0,
        y: jax.Array | float = 0.0,
        ud: jax.Array | float,
        spec_shape: SpectralShape,
    ) -> None:
        r"""**Arguments**
        - `x`: The $x$-position of the source in $\mathrm{mas}$.
        - `y`: The $y$-position of the source in $\mathrm{mas}$.
        - `ud`:  The angular uniform disk size in $\mathrm{mas}$.
        - `spec_shape`: The relative spectral shape of the geometric component.
        """
        self.x = x
        self.y = y
        self.ud = ud
        self.spec_shape = spec_shape

    def get_complex_visibilities(self, u: jax.Array, v: jax.Array) -> jax.Array:
        r"""Compute complex visibilities (including component's position).

        **Arguments**

        - `u`: 1D array of $u$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.
        - `v`: 1D array of $v$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.

        **Returns**

        A complex 1D array holding the complex visibilities calculated at the specified
        $uv$-coordinates.
        """
        raise NotImplementedError


class PointSource(GeometricComponent):
    r"""A point source.

    **Attributes**

    - `x`: The $x$-position of the source in $\mathrm{mas}$.
    - `y`: The $y$-position of the source in $\mathrm{mas}$.
    - `spec_shape`: The relative spectral shape of the geometric component.

    The attributes can be initialised as either JAX arrays, in which case they are free
    to be optimised, or floats, in which case they are considered static."""

    # Required instance attributes.
    x: jax.Array | float
    y: jax.Array | float
    spec_shape: SpectralShape

    def __init__(
        self, *, x: float | jax.Array, y: float | jax.Array, spec_shape: SpectralShape
    ) -> None:
        r"""**Arguments**
        - `x`: The $x$-position of the source in $\mathrm{mas}$.
        - `y`: The $y$-position of the source in $\mathrm{mas}$.
        - `spec_shape`: The relative spectral shape of the geometric component.
        """
        self.x = x
        self.y = y
        self.spec_shape = spec_shape

    def get_complex_visibilities(self, u: jax.Array, v: jax.Array) -> jax.Array:
        r"""Compute complex visibilities (including component's position).

        **Arguments**

        - `u`: 1D array of $u$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.
        - `v`: 1D array of $v$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.

        **Returns**

        A complex 1D array holding the complex visibilities calculated at the specified
        $uv$-coordinates.
        """
        # Component centered visibility
        vis = jnp.array(1)
        # Component visibility including positional offset
        vis = self._apply_positional_phase_offset(vis, u, v)
        return vis


class PowerLaw(SpectralShape):
    r"""A flux power law.

    **Attributes**

    - `d`: The spectral index $d$ such that the $F_{\lambda}$ profile follows
        $\propto \lambda^{d}$.
    """

    # Required instance attributes.
    d: float | jax.Array

    def __init__(self, d: float | jax.Array) -> None:
        r"""**Arguments**
        - `d`: The spectral index $d$ such that the $F_{\lambda}$ profile follows
        $\propto \lambda^{d}$.
        """
        self.d = d

    def get_flux(
        self, wavelengths: jax.Array, *, wave0: float, f0: float | jax.Array
    ) -> jax.Array:
        r"""Retrieve the $F_{\lambda}$ flux at desired wavelengths given a reference
        wavelength and reference flux.

        **Arguments**

        - `wavelengths`: JAX array containing the wavelengths in $\mathrm{m}$ at which
            to calculate the spectral flux.
        - `wave0`: The reference wavelength in $\mathrm{m}$. We define
            the reference flux `f0` at this point, with the returned flux being
            calculated relative to these reference values.
        - `f0`: The reference $F_{\lambda}$ flux level.

        **Returns**

        A 1D array containing the output $F_{\lambda}$ spectrum according to the
        spectral shape. This is calculated assuming a reference $F_{\lambda}$ flux `f0`
        at a reference wavelength `wave0`. The units of this will be whatever the
        original units of `f0` are (which can also represent flux fractions).
        """
        return f0 * (wavelengths / wave0) ** self.d


class BlackBody(SpectralShape):
    r"""A blackbody spectral law.

    **Attributes**

    - `temperature`: The blackbody's temperature in $\mathrm{K}$.
    """

    temperature: float | jax.Array

    def __init__(self, temperature: float | jax.Array) -> None:
        r"""**Arguments**

        - `temperature`: The blackbody's temperature in $\mathrm{K}$.
        """
        self.temperature = temperature

    def get_flux(
        self, wavelengths: jax.Array, *, wave0: float, f0: float | jax.Array
    ) -> jax.Array:
        r"""Retrieve the $F_{\lambda}$ flux at desired wavelengths given a reference
        wavelength and reference flux.

        **Arguments**

        - `wavelengths`: JAX array containing the wavelengths in $\mathrm{m}$ at which
            to calculate the spectral flux.
        - `wave0`: The reference wavelength in $\mathrm{m}$. We define
            the reference flux `f0` at this point, with the returned flux being
            calculated relative to these reference values.
        - `f0`: The reference $F_{\lambda}$ flux level.

        **Returns**

        A 1D array containing the output $F_{\lambda}$ spectrum according to the
        spectral shape. This is calculated assuming a reference $F_{\lambda}$ flux `f0`
        at a reference wavelength `wave0`. The units of this will be whatever the
        original units of `f0` are (which can also represent flux fractions).
        """
        raise NotImplementedError
