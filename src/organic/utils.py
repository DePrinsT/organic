"""Module with various utilities that are used at multiple points in the package, or
which are useful as standalone components in interpreting data or analysing
Organic's outputs."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp
from jaxtyping import PyTree

from ._consts import MAS2RAD

# --- IMAGE AND FFT CALCULATION UTILITIES ---


@jax.jit(static_argnames="ps")
def img_get_sky_coordinates(img: jax.Array, *, ps: float):
    r"""Calculate the interferometric sky coordinates from a 2D image according to
    interferometric convention (positive x is towards the left, positive y towards the
    top). Note that this returns the coordinates of the centers of the pixels, not of
    the edges. This works for both even and uneven (or mixed) amounts of pixels along
    the axes, assuming the origin is located at the geometric center of the image.

    **Arguments**

    - `img`: The image to calculate coordinates for. Should be a 2D image.
    - `ps`: The pixelscale of the image in $\mathrm{mas}$.

    **Returns**

    - `x`: A 1D array with the x-coordinates in $\mathrm{mas}$.
    - `y`: A 1D array with the x-coordinates in $\mathrm{mas}$."""
    # NOTE: take care with the dimension axis convention of numpy versus that of optical
    # interferometry. For numpy, the y direction is the first index, the x direction
    # the second.
    nx, ny = img.shape[1], img.shape[0]  # number of pixels

    # Calculate on-sky coordinates of pixel centers.

    # NOTE: take care with the coordinate convention of optical interferometry. The
    # x-coordinate is defined from right to left in the image (from west to east), and
    # the y-coordinate from bottom to top (south to north). The formulation below works
    # for both uneven and even amounts of pixels in either dimension (one can also be
    # even and the other uneven).
    j = jnp.arange(nx)
    i = jnp.arange(ny)

    x = -(j - (nx - 1) / 2) * ps
    y = ((ny - 1) / 2 - i) * ps

    return x, y


@jax.jit(static_argnames=["padding"])
def img_get_complex_vis_fft(
    img: jax.Array,
    u: jax.Array,
    v: jax.Array,
    *,
    ps: float,
    padding: tuple[int, int] | None = None,
):
    r"""Uses FFT to calculate complex visibilities for an input 2D image (essentially
    assuming a delta-function pulse response for each pixel).

    The FFT values are mapped onto the reqauired $uv$-coordinates using bilinear
    interpolation. This implementation is likely only trustworthy to about $10^{-6}$
    degrees in the resulting phase (if sufficient padding is provided).

    **Arguments**

    - `img`: The 2D input image array.
    - `u`: 1D array of $u$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.
    - `v`: 1D array of $u$-coordinates in the Fourier plane in $\mathrm{rad^{-1}}$.
    - `ps`: Pixelscale of the image in $\mathrm{mas / pix}$.
    - `padding`: Amount of pixels to which the image should be zero-padded before
        performing the FFT. I.e. `padding = (1024, 512)` will pad the image to
        1024 and 512 in the $x$ and $y$ direction respectively. Note the padding will
        only be applied in the respective direction if the padded size is larger than
        the original size. This will increase the relative resolution in spatial
        frequency space.

    **Returns**

    A complex 1D array holding the complex visibilities calculated at the specified
    $uv$-coordinates.
    """
    # Unit conversions: pixelscale in radians for further use.
    ps_rad = ps * MAS2RAD

    # NOTE: take care with the dimension axis convention of Numpy versus that of optical
    # interferometry. For Numpy, the y direction is the first, the x direction is the
    # second.
    nx, ny = img.shape[1], img.shape[0]  # number of pixels

    # -- Perform padding --

    if padding is None:
        img_padded = img
    else:
        # NOTE: for uneven amount of padding pixels to be added, the excess pixels will
        # be added at the end of the array axes.
        npad_x = 0 if (padding[0] - nx < 0) else padding[0] - nx
        npad_y = 0 if (padding[1] - ny < 0) else padding[1] - ny

        # Zero-pad the image.
        img_padded = jnp.pad(
            img,
            pad_width=(
                (npad_y // 2, (npad_y // 2) + npad_y % 2),
                (npad_x // 2, (npad_x // 2) + npad_x % 2),
            ),
            mode="constant",
            constant_values=0.0,
        )

    # -- Perform FFT --

    # Numpy FFT frequencies are in cycles per unit of input spacing.
    # NOTE: pixelscale used is in radian to get spatial frequencies in rad^-1
    nx_padded = img_padded.shape[1]
    ny_padded = img_padded.shape[0]
    # Get spatial frequencies and put in ascending order (using fftshift).
    # NOTE: to follow the interferometric convention, we should put a minus sign here
    # (the direction of the axes that fftfreq assumes is reversed), yet JAX
    # RegularGridInterpolator does not accept descending order points for now.
    # We'll account for this when passing our uv-points to the interpolator later on.
    u_grid = np.fft.fftshift(np.fft.fftfreq(nx_padded, d=ps_rad))
    v_grid = np.fft.fftshift(np.fft.fftfreq(ny_padded, d=ps_rad))

    # NOTE: due to the Numpy conventions (which assumes the phase-center is at the top-
    # left pixel's center), you must first shift the image in order to get the phase-
    # center to be defined in the middle of the image. This esentially swaps image
    # quadrants so the phase reference center is at the geometric center of the image
    # in case of uneven amount of pixels, and half a pixel offset to the right and
    # bottom w.r.t. the actual geometric center (which now at the vertex between the
    # central few pixels). We correct for potential phase offsets induced by this
    # w.r.t. to the geometric center of the original image further on.
    img_shifted = jnp.fft.ifftshift(img_padded)  # Shift quadrants.
    fft_img = jnp.fft.fftshift(
        jnp.fft.fft2(img_shifted)
    )  # Re-order frequency-space output to go from negative to positive frequencies.

    # NOTE: for the scipy interpolator, JAX has no out-of-bounds error, adding NaNs
    # instead. Before running this function, caller should check beforehand if all
    # the uv-points will be covered.
    interpolator = jsp.interpolate.RegularGridInterpolator(
        (v_grid, u_grid),
        fft_img,
    )

    # NOTE: because the JAX RegularGridInterpolator only accepts ascending order points
    # for now, we have to account for the fact that the sign of the spatial frequencies
    # the interpolator expects is flipped w.r.t. interferometric convention (by adding
    # minuses to the interferometric uv-points we pass to the interpolator).
    points = jnp.column_stack((-v, -u))
    vis = interpolator(points)

    # NOTE: implement correction for phase center offset so the phase center is always
    # in the geometric center of the original image (this depends on the evenness/
    # oddness of the amount of pixels in the original image and the padded image).
    # No correction is needed if going from uneven original to uneven padded images.

    # For x-axis
    if (nx % 2 == 0) and (nx_padded % 2 == 0):  # Even-to-even
        vis *= jnp.exp(2j * jnp.pi * (ps_rad / 2) * u)
    elif (nx % 2 == 0) and (nx_padded % 2 == 1):  # Even-to-uneven
        vis *= jnp.exp(2j * jnp.pi * (ps_rad / 2) * u)
    elif (nx % 2 == 1) and (nx_padded % 2 == 0):  # Uneven-to-even
        vis *= jnp.exp(2j * jnp.pi * ps_rad * u)
    # For y-axis
    if (ny % 2 == 0) and (ny_padded % 2 == 0):  # Even-to-even
        vis *= jnp.exp(2j * jnp.pi * (ps_rad / 2) * v)
    elif (ny % 2 == 0) and (ny_padded % 2 == 1):  # Even-to-uneven
        vis *= jnp.exp(2j * jnp.pi * (ps_rad / 2) * v)
    elif (ny % 2 == 1) and (ny_padded % 2 == 0):  # Uneven-to-even
        vis *= jnp.exp(2j * jnp.pi * ps_rad * v)

    return vis


# --- JAX-RELATED UTILS ---


def _tree_print_keypaths(
    tree: PyTree,
    show_vals: bool = False,
    show_treedef: bool = False,
    show_fullkey: bool = False,
) -> None:
    """Sequentially prints out the keypaths and types of leaves in a pytree.

    **Arguments:**

    - `tree`: Any pytree.
    - `show_vals`: Whether to show the leaf value.
    - `show_treedef`: Whether to also show the JAX pytree definition up top.
    - `show_fullkey`: whether to show the keypath to the leaves.

    **Returns:**

    Nothing.
    """
    line_sep = "=" * 56
    flattened, treedef = jax.tree_util.tree_flatten_with_path(tree)
    if show_treedef:
        print(f"{line_sep}\nTREE DEFINITION: {treedef}\n{line_sep}\n\n")
    print(f"PYTREE'S TYPE: {type(tree)}\n{line_sep}\n")
    for key_path, value in flattened:
        print(f"LEAF KEYPATH: {jax.tree_util.keystr(key_path)}")
        print(f"PYTHON TYPE: {type(value)}")
        if show_vals:
            print(f"LEAF VALUE:\n{value}")
        if show_fullkey:
            print(f"FULL KEYPATH: {key_path}")
        print(line_sep)
    return


# --- MISCELLANEOUS ---


def generate_ring_img(
    npix: int,
    diam: float,
    ps: float,
    width_frac: float,
) -> np.ndarray:
    r"""Generate a 2D image of a thin ring with a given width.

    **Parameters**

    - `npix`: Number of pixels along one axis (image is square).
    - `diam`: Ring outer diameter in $\mathrm{mas}$.
    - `ps`: Pixelscale in $\mathrm{mas / pix}$.
    - `width_frac`: Fraction of the radius for the ring thickness.

    **Returns**

    2D array containing the ring image.
    """
    # Convert diameter to radius in pixels
    radius_pix = (diam / 2) / ps
    width_pix = radius_pix * width_frac

    # Coordinate grids centred at middle of image
    y, x = np.indices((npix, npix)) - npix / 2
    r = np.sqrt(x**2 + y**2)

    # Ring mask: pixels within radius ± half-width
    mask = np.logical_and(
        r >= radius_pix - width_pix / 2, r <= radius_pix + width_pix / 2
    )

    # Create image
    img = np.zeros((npix, npix), dtype=np.float32)
    img[mask] = 1.0  # Set ring intensity

    return img


def get_thin_ring_null(diam: float, n: int) -> float:
    r"""Compute the spatial frequency corresponding to the nth null of a thin ring's
    visibility profile.

    **Arguments**

    - `diam`: Outer diameter of the ring in $\mathrm{mas}$.
    - `n`: Number of nulls to calculate (i.e. `n=1` gives only the first null position).

    **Returns**

    First null spatial frequency in $\mathrm{rad^{-1}}$.
    """
    # Convert outer diameter to radius in radians
    radius_mas = diam / 2
    radius_rad = radius_mas * MAS2RAD  # mas -> rad

    # nth zero of Bessel J0 function.
    zero_point = sp.special.jn_zeros(0, n)[n - 1]

    # Spatial frequency in cycles/rad.
    f_rad = zero_point / (2 * np.pi * radius_rad)

    return f_rad
