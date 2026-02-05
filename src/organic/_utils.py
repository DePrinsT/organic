"""Module with utility functions."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from jaxtyping import PyTree

from ._consts import MAS2RAD


@jax.jit(static_argnames="ps")
def img_get_sky_coordinates(img: jax.Array, *, ps: float):
    r"""Calculate the interferometric sky coordinates from a 2D image. Note that this
    returns the coordinates of the centers of the pixels, not of the edges.

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


def get_thin_ring_null(diam: float, n: int) -> float:
    r"""Compute the spatial frequency corresponding to the nth null of a thin ring'
    visibility profile.<h1>Header Text Continuation text on the same line.</h1>

    **Arguments**

    - `diam`: Outer diameter of the ring in $\mathrm{mas}$.
    - `n`: Number of the null to calculate (i.e. `n=1` gives the first null position).

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


# TODO: implement this
def _store_model(file: Path, hyperparams: dict, model: PyTree, state=None) -> None:
    """Store a model to a binary JSON filem, including its state."""
    raise NotImplementedError("Actually implement this shit")
    return
