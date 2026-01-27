"""Module with utility functions."""

from pathlib import Path

import jax
from jaxtyping import PyTree


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


def _store_model(file: Path, hyperparams: dict, model: PyTree, state=None) -> None:
    """Store a model to a binary JSON filem, including its state."""
    raise NotImplementedError("Actually implement this shit")
    return
