"""Module for training WGANs for use in image reconstruction. Training is started
and controlled using the master function [train_wgan]
[organic.model_training.train_wgan]."""

import functools as ft
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from jaxtyping import PyTree
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from organic.utils import ORGANIC_ASCII_ART, _create_output_dir

# TODO: if in inference mode, model states are not updated using e.g. state.set
# by equinox, meaning you don't need to thread the state through for that model.
# Hence can make the code here clearer (and perhaps less bug-prone by preventing
# unintended model state updates) by removing state return values and threading if
# the corresponding model is expected to be in infernece mode.

# TODO: make sure all functions are properly documented


class WGANCallable(Protocol):
    """Protocol defining the call signature that ORGANIC expects for a WGAN generator
    or critic component. Specifically, ORGANIC expects a registered PyTree with the
    following call signature: `__call__(self, x: jax.Array, state: eqx.nn.State, *,
    key: jax.Array) -> tuple[jax.Array, eqx.nn.State]`. This does not mean that
    components must actually implement stateful or stochastic layers (via the `key`
    argument), but the `state` and `key` arguments must be present for general API
    compatibility with ORGANIC's internals. This means that any kind of PyTree passed
    to ORGANIC's API must be initialized via `equinox.nn.make_with_state` (this works
    even if the PyTree does not inherit from `equinox.Module`).
    """

    def __call__(
        self, x: jax.Array, state: eqx.nn.State, *, key: jax.Array
    ) -> tuple[jax.Array, eqx.nn.State]: ...


# Type for a WGAN component, which is just expected to be a PyTree with `__call__`
# signature specified by the `WGANCallable` protocol.
type WGANComponent = PyTree[WGANCallable]
"""Type for a PyTree with the `__call__` signature of the `WGANCallable` protocol.
"""


def train_wgan(
    gen_params: WGANComponent,
    gen_static: WGANComponent,
    gen_state: eqx.nn.State,
    crit_params: WGANComponent,
    crit_static: WGANComponent,
    crit_state: eqx.nn.State,
    opt_gen: optax.GradientTransformation,
    opt_gen_state: optax.OptState,
    opt_crit: optax.GradientTransformation,
    opt_crit_state: optax.OptState,
    training_loader: Iterable[jax.Array],
    *,
    output_dir: str | os.PathLike[str],
    ngen: int,
    ncrit_ratio: int,
    key: jax.Array,
    size_in: int = 0,
    ncheck: int | None = None,
    override: bool = False,
    show_progress_bar: bool = False,
) -> tuple[
    WGANComponent,
    WGANComponent,
    eqx.nn.State,
    WGANComponent,
    WGANComponent,
    eqx.nn.State,
    optax.OptState,
    optax.OptState,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Trains a WGAN consisting of a generator and a critic using the Wasserstein
    metric under the Kantorovich-Rubinstein duality (minimization of the critic).

    This function is meant to be compatible with the use of any callable (batch-aware)
    stateful Equinox modules for the generator and critic. Note that this does not mean
    these modules need to actually use any internal state (i.e. actively use it in their
    internal `__call__` function), they just need to be initialized using
    `eqx.nn.make_with_state()` to provide the relevant state (even if just a dummy)
    and have a general call signature of the form `__call__(x, state, *, key)`.
    Note that if you use calculations in the model components that need to be
    batch-aware (e.g. `jax.lax.psum`), then this function assumes `axis_name = "batch"`.

    **Arguments**

    - `gen_params`: The WGAN generator. This should be a callable PyTree that produces
        a 3D image (Channel, Y, X indexing) cube from an input 1D latent vector. This
        parameter contains the trainable parameters of the generator.
    - `gen_static`: The static components of the generator, which are considered fixed
        during training and are used as hashable static arguments for JAX JITed training
        steps. Note that when checkpointing this is stored in inference mode.
    - `gen_state`: Initial state of the generator before training commences.
    - `crit_params`: The WGAN critic. This should be a callable PyTree that produces a
        scalar from a 3D image (Channel, Y, X indexing) cube. This parameter contains
        the trainable parameters of the generator.
    - `crit_static`: The static components of the critic, which are considered fixed
        during training and are used as hashable static arguments for JAX JITed training
        steps. Note that when checkpointing this is stored in inference mode.
    - `crit_state`: Initial state of the critic before training commences.
    - `opt_gen`: Optax optimizer for the generator.
    - `opt_gen_state`: State of the Optax optimizer for the generator.
    - `opt_crit`: Optax optimizer for the critic.
    - `opt_gen_state`: State of the Optax optimizer for the critic.
    - `training_loader`: Image loader Ierable returning batches of the training data in
        4D arrays using (Batch, Channel, Y, X) index ordering.
    - `output_dir`: Directory in which to store outputs from the training procedure,
        including additional diagnostics if asked for.
    - `ngen`: Number of generator training steps to take.
    - `ncrit_ratio`: Ratio of critic training steps per generator training steps
        (should ideally be `>= 5` to ensure the critic remains close to optimality).
    - `key`: JAX PRNG key used for e.g. generator latent random input vector
        generation.
    - `ncheck`: Make checkpoint serializations of the model and the optimizers
        at the end of every `ncheck` optimization steps of the `gen` generator.
        Note that a checkpoint is always made at the very end of the optimization
        by default.
    - `size_in`: Generator input vector size. This does not need to be specified if
        the passed along generator `gen` already has a `size_in` attribute, which
        takes priority. In case of the latter, be sure to mark this attribute as static
        with `eqx.field(static=True)`.
    - `override`: Whether to override the contents of `output_dir`. Note that if this
        is set to `override = True` it will all delete the contents in `output_dir`
        if it already exists.
    - `show_progress_bar`: Whether to show a tqdm progress bar. This is useful to
        track generator update steps if you are running this function with an output
        terminal. The progress bar is best not used in situations where terminal
        output is piped to a log file, since the carrion returns used by tqdm
        will not be interpreted properly, which can mess up formating.
    return (
        gen_params,
        gen_static,
        gen_state,
        crit_params,
        crit_static,
        crit_state,
        opt_gen_state,
        opt_crit_state,
        gen_losses,
        crit_losses,
        scores_training_imgs,
        scores_gen_imgs,
    )
    **Returns**

    Returns the updated values of `gen_params`, `gen_static`, `gen_state`,
    `crit_params`, `crit_static`, `crit_state`, `opt_gen_state` and `opt_crit_state`.
    In addition, returns the generator and critic losses as well as the individual
    mean critic scores of the training and generated images.

    TODO: describe the returned values.
    """
    # Check if training step values make sense.
    if ngen < 1 or ncrit_ratio < 1:
        raise ValueError(
            "'ngen' and 'ncrit_ratio' must be larger than one. Current"
            f"values are 'ngen' = {ngen} & 'ncrit_ratio' = {ncrit_ratio}."
        )

    # Greeting message.
    print(ORGANIC_ASCII_ART)
    str_greet = "Starting WGAN training"
    print(f"{'=' * len(str_greet)}\n{str_greet}\n{'=' * len(str_greet)}")

    # --- Create output directory ---
    output_dir = Path(output_dir)
    _create_output_dir(output_dir, override=override)
    # ---

    # --- Checks on dimensionality and shape matching ---
    # Check if `gen` input size has been given either in `gen` or in function arguments.
    size_in = _gen_resolve_latent_size(gen_static, size_in)

    # Initialize training image iterator.
    training_img_iter = iter(training_loader)

    key, subkey1, subkey2 = jr.split(key, 3)
    # Check generator and critic output dimensionality and compatibility. Have to
    # put in inference mode so the states don't get invalidated by Equinox.
    gen_static = eqx.nn.inference_mode(gen_static, value=True)
    crit_static = eqx.nn.inference_mode(crit_static, value=True)
    _wgan_check_gen_and_crit(
        eqx.combine(gen_params, gen_static),
        gen_state,
        eqx.combine(crit_params, crit_static),
        crit_state,
        size_in=size_in,
        key=subkey1,
    )
    # Check generator and training image batch dimensionality and compatibility.
    batch_size = _wgan_check_gen_and_training_loader(
        eqx.combine(gen_params, gen_static),
        gen_state,
        training_img_iter,
        size_in=size_in,
        key=subkey2,
    )
    # ---

    # --- Main training loop ---
    # Track loss values and scores through training.
    gen_loss_list = []  # Generator loss after a generator update.
    crit_loss_list = []  # Critic ross right before a generator update.
    score_training_imgs_list = []  # Training image critic score.
    score_gen_imgs_list = []  # Generated image critic score.

    # Loop over generator training steps.
    for i_gen in tqdm(
        range(1, ngen + 1), desc="Generator updates", disable=not show_progress_bar
    ):
        # Inernal loop over critic training steps.
        crit_loss, score_training_imgs, score_gen_imgs = 0, 0, 0
        for _ in range(1, ncrit_ratio + 1):
            # Set generator in inference mode
            gen_static = eqx.nn.inference_mode(gen_static, value=True)
            # Set critic in training mode
            crit_static = eqx.nn.inference_mode(crit_static, value=False)
            # Get training image batch
            training_img_batch = next(training_img_iter)
            # Perform critic training step on `crit_params`.
            key, subkey = jr.split(key, 2)
            (
                crit_params,
                crit_state,
                gen_state,
                opt_crit_state,
                crit_loss,
                score_training_imgs,
                score_gen_imgs,
            ) = _wgan_crit_make_step(
                crit_params,
                crit_static,
                crit_state,
                gen_params,
                gen_static,
                gen_state,
                opt_crit,
                opt_crit_state,
                key=subkey,
                size_in=size_in,
                batch_size=batch_size,
                training_img_batch=training_img_batch,
            )
        # Track critic loss. and scores
        crit_loss_list.append(float(crit_loss))
        score_training_imgs_list.append(float(score_training_imgs))
        score_gen_imgs_list.append(float(score_gen_imgs))
        # Set generator in training mode
        gen_static = eqx.nn.inference_mode(gen_static, value=False)
        # Set critic in inference mode
        crit_static = eqx.nn.inference_mode(crit_static, value=True)
        # Perform generator training step on `gen_params`.
        key, subkey = jr.split(key, 2)
        gen_params, gen_state, crit_state, opt_gen_state, gen_loss = (
            _wgan_gen_make_step(
                gen_params,
                gen_static,
                gen_state,
                crit_params,
                crit_static,
                crit_state,
                opt_gen,
                opt_gen_state,
                key=subkey,
                size_in=size_in,
                batch_size=batch_size,
            )
        )
        # Track generator loss.
        gen_loss_list.append(float(gen_loss))

        # Make checkpoint.
        if ncheck is not None and (i_gen % ncheck == 0):
            # Set model components in inference mode before checkpointing.
            gen_static = eqx.nn.inference_mode(gen_static, value=True)
            crit_static = eqx.nn.inference_mode(crit_static, value=True)
            # Create specific directory to save this checkpoint in.
            checkpoint_dir = output_dir / f"checkpoint_N{i_gen}"
            _create_output_dir(checkpoint_dir, override=override)
            _wgan_training_make_checkpoint(
                checkpoint_dir,
                gen_params,
                gen_state,
                crit_params,
                crit_state,
                opt_gen_state,
                opt_crit_state,
            )
            # Make plot of generator-drawn images.
            key, subkey = jr.split(key, 2)
            _gen_plot_image_examples(
                gen_params,
                gen_static,
                gen_state,
                filename=checkpoint_dir / "gen_sample_imgs.pdf",
                nimg=25,
                size_in=size_in,
                key=subkey,
            )
            # Make plots of loss functions and critic scores.
            _wgan_training_store_loss_trajectories(
                np.array(crit_loss_list),
                np.array(gen_loss_list),
                np.array(score_training_imgs_list),
                np.array(score_gen_imgs_list),
                out_dir=checkpoint_dir,
            )

    # ---

    # --- Final checkpoint and saving of outputs at the end of training loop ---
    # Make checkpoint.
    # Set model components in inference mode before checkpointing.
    gen_static = eqx.nn.inference_mode(gen_static, value=True)
    crit_static = eqx.nn.inference_mode(crit_static, value=True)
    # Create specific directory to save this checkpoint in.
    checkpoint_dir = output_dir / "checkpoint_final"
    _create_output_dir(checkpoint_dir, override=override)
    _wgan_training_make_checkpoint(
        checkpoint_dir,
        gen_params,
        gen_state,
        crit_params,
        crit_state,
        opt_gen_state,
        opt_crit_state,
    )
    # Make plot of generator-drawn images.
    key, subkey = jr.split(key, 2)
    _gen_plot_image_examples(
        gen_params,
        gen_static,
        gen_state,
        filename=checkpoint_dir / "gen_sample_imgs.pdf",
        nimg=25,
        size_in=size_in,
        key=subkey,
    )
    # Make plots of loss functions and critic scores.
    _wgan_training_store_loss_trajectories(
        np.array(crit_loss_list),
        np.array(gen_loss_list),
        np.array(score_training_imgs_list),
        np.array(score_gen_imgs_list),
        out_dir=checkpoint_dir,
    )
    # ---

    # Finished message.
    str_finish = f"Training finished after {ngen} generator updates"
    print(f"{str_finish}\n{'=' * len(str_finish)}")

    # Return.
    gen_losses = np.array(gen_loss_list)
    crit_losses = np.array(crit_loss_list)
    scores_training_imgs = np.array(score_training_imgs_list)
    scores_gen_imgs = np.array(score_gen_imgs_list)

    return (
        gen_params,
        gen_static,
        gen_state,
        crit_params,
        crit_static,
        crit_state,
        opt_gen_state,
        opt_crit_state,
        gen_losses,
        crit_losses,
        scores_training_imgs,
        scores_gen_imgs,
    )


def _gen_plot_image_examples(
    gen_params: WGANComponent,
    gen_static: WGANComponent,
    gen_state: eqx.nn.State,
    *,
    filename: str | os.PathLike[str],
    nimg: int,
    size_in: int,
    key: jax.Array,
    dpi: int = 200,
) -> eqx.nn.State:
    """Function to plot example images drawn from the generator in rows of five. Note
    that the images are assumed to be normalized in the range $[-1, 1]$.

    **Arguments**

    - `gen_params`: The WGAN generator. This should be a callable PyTree that produces
        a 3D image (Channel, Y, X indexing) cube from an input 1D latent vector. This
        parameter contains the trainable parameters of the generator.
    - `gen_static`: The static components of the generator, which are considered fixed
        during training and are used as hashable static arguments for JAX JITed training
        steps. Note that when checkpointing this is stored in inference mode.
    - `gen_state`: Initial state of the generator before training commences.
    - `filename`: File to save the plot of the sampled images to.
    - `nimg`: Number of images to plot.
    - `size_in`: Size of generator input latent vector.
    - `key`: JAX PRNG key.
    - `dpi`: DPI of saved plot

    **Returns**

    The generator state in `gen_state`. It should not have been changed in any way
    (see the warning below). It is returned for API compatibility with Equinox.

    !!! warning

        Note that the generator should be set in inference mode before calling this
        function. Its state should not be updated. This function trusts the caller made
        sure of this. This function should not be used during training, and is only
        meant to produce diagnostic images.
    """
    # Retrieve batch of images.
    gen_img_batch, gen_state = _gen_get_image_examples(
        gen_params, gen_static, gen_state, nimg=nimg, size_in=size_in, key=key
    )
    # Make plots.
    ncols = 5
    nrows = (nimg + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes_flat = axes.flat
    # Plot images.
    for i in range(nimg):
        im = axes_flat[i].imshow(gen_img_batch[i, 0, :, :], vmin=-1, vmax=1)
        # Add colorbar
        if i == nimg - 1:
            divider = make_axes_locatable(axes_flat[i])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, cax=cax)
    # Delete superfluous axes.
    for ax in axes_flat[nimg - 1 :]:
        fig.delaxes(ax)
    # Save figure.
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return gen_state


@ft.partial(jax.jit, static_argnames=("gen_static", "nimg", "size_in"))
def _gen_get_image_examples(
    gen_params: WGANComponent,
    gen_static: WGANComponent,
    gen_state: eqx.nn.State,
    *,
    nimg: int,
    size_in: int,
    key: jax.Array,
) -> tuple[jax.Array, eqx.nn.State]:
    """Function to retrieve a batch of random images drawn from the generator.

    **Arguments**

    - `gen_params`: The WGAN generator. This should be a callable PyTree that produces
        a 3D image (Channel, Y, X indexing) cube from an input 1D latent vector. This
        parameter contains the trainable parameters of the generator.
    - `gen_static`: The static components of the generator, which are considered fixed
        during training and are used as hashable static arguments for JAX JITed training
        steps. Note that when checkpointing this is stored in inference mode.
    - `gen_state`: Initial state of the generator before training commences.
    - `filename`: File to save the plot of the sampled images to.
    - `nimg`: Number of images to plot.
    - `size_in`: Size of generator input latent vector.
    - `key`: JAX PRNG key.
    - `dpi`: DPI of saved plot

    **Returns**

    The resulting image batch in a 4D JAX array `gen_img_batch` (using Batch, Channel,
    Y, X indexing). The generator state in `gen_state`. The generator state should not
    have been changed in any way (see the warning below). It is returned for API
    compatibility with Equinox.

    !!! warning

        Note that the generator should be set in inference mode before calling this
        function. Its state should not be updated. This function trusts the caller made
        sure of this. This function should not be used during training, and is only
        meant to help in producing diagnostic images.
    """
    gen = eqx.combine(gen_params, gen_static)
    subkey1, subkey2 = jr.split(key, 2)
    z_in_batch = jr.normal(subkey1, shape=(nimg, size_in))
    subkeys_gen_batch = jr.split(subkey2, nimg)
    gen_img_batch, gen_state = jax.vmap(gen, in_axes=(0, None), out_axes=(0, None))(
        z_in_batch, gen_state, key=subkeys_gen_batch
    )
    return gen_img_batch, gen_state


def _wgan_training_store_loss_trajectories(
    crit_losses: np.ndarray,
    gen_losses: np.ndarray,
    scores_training_imgs: np.ndarray,
    scores_gen_imgs: np.ndarray,
    *,
    out_dir: str | os.PathLike[str],
    dpi: int = 200,
) -> None:
    """Store and make plots of the training losses and critic scores across generator
    updates. The numerical arrays of the losses and scores are stored as Numpy files."""
    # Store loss values to numpy array files.
    out_dir = Path(out_dir)
    np.save(out_dir / "wasserstein_estimate.npy", -crit_losses)
    np.save(out_dir / "generator_loss.npy", gen_losses)
    np.save(out_dir / "critic_score_training_images.npy", scores_training_imgs)
    np.save(out_dir / "critic_score_generator_images.npy", scores_gen_imgs)

    # Make plots.
    fig, axes = plt.subplots(3, 1, figsize=(6, 9))
    axes[0].plot(range(crit_losses.size), -crit_losses)
    axes[0].set_xlabel("Number of generator updates")
    axes[0].set_ylabel("Critic Wasserstein estimate")

    axes[1].plot(range(gen_losses.size), gen_losses)
    axes[1].set_xlabel("Number of generator updates")
    axes[1].set_ylabel("Generator loss")

    axes[2].plot(
        range(scores_training_imgs.size), scores_training_imgs, label="training images"
    )
    axes[2].plot(
        range(scores_training_imgs.size), scores_gen_imgs, label="generated images"
    )
    axes[2].set_xlabel("Number of generator updates")
    axes[2].set_ylabel("Critic batch score")
    axes[2].legend()
    fig.tight_layout()
    # Save figure.
    fig.savefig(out_dir / "loss_trajectories.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return


def _wgan_training_make_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    gen_params,
    gen_state,
    crit_params,
    crit_state,
    opt_gen_state,
    opt_crit_state,
) -> None:
    """Serialize the trainable parameters and state for the WGAN generator, critic
    and their corresponding optimizer states."""
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "gen_params.eqx", "wb") as f:
        eqx.tree_serialise_leaves(f, gen_params)
        with open(checkpoint_dir / "gen_state.eqx", "wb") as f:
            eqx.tree_serialise_leaves(f, gen_state)
        with open(checkpoint_dir / "crit_params.eqx", "wb") as f:
            eqx.tree_serialise_leaves(f, crit_params)
        with open(checkpoint_dir / "crit_state.eqx", "wb") as f:
            eqx.tree_serialise_leaves(f, crit_state)
        with open(checkpoint_dir / "opt_gen_state.eqx", "wb") as f:
            eqx.tree_serialise_leaves(f, opt_gen_state)
        with open(checkpoint_dir / "opt_crit_state.eqx", "wb") as f:
            eqx.tree_serialise_leaves(f, opt_crit_state)
    return


def _gen_resolve_latent_size(gen_static: WGANComponent, size_in: int) -> int:
    """Resolve the size of the 1D latent space vector used in the generator. This is
    done by giving priority to the attribute value given in the generator's static
    component. Otherwise, the `size_in` argument passed to the master training function
    (e.g. `train_wgan`) is used.

    **Arguments**

    - `gen_static`: The static component of the generator.
    - `size_in`: The latent size input argument passed to the master training function.

    **Returns**

    The size of the 1D latent vector space.
    """
    if hasattr(gen_static, "size_in"):
        size_in = getattr(gen_static, "size_in")
        print(
            f"Latent generator input size derived from 'size_in' attribute: {size_in}"
        )
    elif size_in <= 0:
        raise ValueError(
            "Argument 'size_in' is <=0 and generator has no 'size_in' attribute."
            "Cannot infer the size of the latent vector space."
            "Please specify either (the generator attribute having priority)."
        )
    else:
        size_in = size_in
        print(f"Latent generator input size derived from 'size_in' argument: {size_in}")
    return size_in


def _wgan_check_gen_and_crit(
    gen: WGANComponent,
    gen_state: eqx.nn.State,
    crit: WGANComponent,
    crit_state: eqx.nn.State,
    *,
    size_in: int,
    key: jax.Array,
) -> None:
    """Check the shape compatibility of the generator and critic.

    **Arguments**

    - `gen`: The WGAN generator. This should be a callable PyTree that produces either
         a 3D image (Channel, Y, X indexing) cube from an input 1D latent vector.
    - `gen_state`: Current state of the generator.
    - `crit`: The WGAN critic. This should be a callable PyTree that produces a scalar
        from a 3D image (Channel, Y, X indexing) cube.
    - `crit_state`: Current state of the critic.
    - `size_in`: Size of 1D generator input latent vector.
    - `key`: JAX PRNG key used to generate a generator latent random input vector for
        testing.

    !!! warning

        Note that the generator and critic are best set to inference mode before this so
        their states do not get invalidated by the model calls made in this function.
    """
    key, subkey1, subkey2, subkey3 = jr.split(key, 4)
    z_test = jr.normal(key=subkey1, shape=(size_in,))
    img_gen_test, _ = gen(z_test, gen_state, key=subkey2)
    if img_gen_test.ndim != 3:
        raise ValueError(
            "Generator should produce 3D image cubes using "
            "(Channel, Y, X) indexing. Instead found a generator"
            f"output with dimensionality {img_gen_test.ndim}!"
        )
    print(
        "Shape of generator output image cube using (Channel, Y, X) indexing: "
        f"{img_gen_test.shape}"
    )
    w_crit_test, _ = crit(img_gen_test, crit_state, key=subkey3)
    if w_crit_test.shape != (1,):
        raise ValueError(
            "The shape of the WGAN critic output should be (1,), i.e. a scalar."
            f"Instead found shape {w_crit_test.shape}. This is likely due to a"
            f"mismatch between the generator output shape {img_gen_test.shape}"
            "and the critic architecture."
        )

    return


def _wgan_check_gen_and_training_loader(
    gen: WGANComponent,
    gen_state: eqx.nn.State,
    training_img_iter: Iterator[jax.Array],
    *,
    size_in: int,
    key: jax.Array,
) -> int:
    """Check the shape compatibility of the generator and training image loader.

    **Arguments**

    - `gen`: The WGAN generator. This should be a callable PyTree that produces either
         a 3D image (Channel, Y, X indexing) cube from an input 1D latent vector.
    - `gen_state`: Current state of the generator.
    - `training_iter`: An Iterator serving up the batches of training images in 4D
        cubes with (Batch, Channel, Y, X) index ordering.
    - `size_in`: Size of 1D generator input latent vector.
    - `key`: JAX PRNG key used to generate a generator latent random input vector for
        testing.

    **Returns**

    The batch size set by the training image loader.

    !!! warning

        Note that the generator is best set to inference mode before this so its state
        does not get invalidated by the model call made in this function.
    """
    key, subkey1, subkey2 = jr.split(key, 3)
    z_test = jr.normal(key=subkey1, shape=(size_in,))
    img_gen_test, _ = gen(z_test, gen_state, key=subkey2)
    if img_gen_test.ndim != 3:
        raise ValueError(
            "Generator should produce 3D image cubes using "
            "(Channel, Y, X) indexing. Instead found a generator"
            f"output with dimensionality {img_gen_test.ndim}!"
        )

    training_batch_test = next(training_img_iter)
    if training_batch_test.ndim != 4:
        raise ValueError(
            "Training image Iterator should produce 4D image cubes using "
            "(Batch, Channel, Y, X) indexing. Instead found a batch Iterator"
            f"output with dimensionality {training_batch_test.ndim}!"
        )
    print(
        "Shape of training image Iterator batch image cube using (Batch, Channel, Y, X)"
        f" indexing: {training_batch_test.shape}"
    )

    img_training_test = training_batch_test[0, :, :, :]
    if img_training_test.shape != img_gen_test.shape:
        raise ValueError(
            "The shape of the images generated by the generator and the "
            "images in the training batch don't match. The former have "
            f"shape {img_gen_test.shape}, the latter have shape "
            f"{img_training_test.shape}"
        )

    return training_batch_test.shape[0]


def _wgan_gen_loss(
    gen_params: WGANComponent,
    gen_static: WGANComponent,
    gen_state: eqx.nn.State,
    crit_params: WGANComponent,
    crit_static: WGANComponent,
    crit_state: eqx.nn.State,
    z_in: jax.Array,
    *,
    key: jax.Array,
) -> tuple[jax.Array, tuple[eqx.nn.State, eqx.nn.State]]:
    """Get the generator's loss value evaluated on a batch of random 1D latent input
    vectors.

    **Arguments**

    - `gen_params`: The generator's trainable parameters. This should generally only
        include components that return `True` under `eqx.is_array`.
    - `gen_static`: The generator's non-trainable components. This should generally only
        include components that return `False` under `eqx.is_array`.
    - `gen_state`: The generator's state. An updated generator state is returned
        after evaluating the batch.
    - `crit_params`: The critic's trainable parameters. This should generally only
        include components that return `True` under `eqx.is_array`.
    - `crit_static`: The critic's non-trainable components. This should generally only
        include components that return `False` under `eqx.is_array`.
    - `crit_state`: The critic's state. Note that the critic's state should not be
        updated in any way meaningful way, since the critic should be in inference mode
        (see the warning below). It is still returned for API compatibility.
    - `z_in`: The batch of 1D generator random latent input vectors. The first index
        should run over the different batch elements, the second over the vectors
        themselves.
    - `key`: JAX PRNG key for calling stochastic layers.

    **Returns**

    Returns a tuple `(gen_loss, (gen_state, crit_state))`, where `gen_loss` is the
    WGAN generator loss over the batch. The state of the generator and critic are also
    returned as auxiliary data. Note that this loss is value is not fully equivalent
    to the actual Wasserstein distance, since the parts of the WGAN Wasserstein estimate
    that are dependent on the training distribution (which do not contribute to the
    gradients of the Wasserstein distance w.r.t. the generator) are dropped.

    !!! warning

        Note that the critic should be set in inference mode before calling this
        function. Since this function is meant to be used in a JAX JIT-ed update step
        of the generator, it implements no way of checking or enforcing this, and trusts
        the caller made sure of this beforehand.
    """
    # Combine params and static into full models again.
    gen = eqx.combine(gen_params, gen_static)
    crit = eqx.combine(crit_params, crit_static)

    # Generate PRNG keys for possible stochastic layers in model calls.
    keys = jr.split(key, z_in.shape[0] + 1)
    key, subkeys_gen_batch = keys[0], keys[1:]
    keys = jr.split(key, z_in.shape[0] + 1)
    key, subkeys_crit_batch = keys[0], keys[1:]

    # Generate batch of images from the generator. Note that JAX automatically considers
    # input keywords to be batched over axis 0. This means we need to provide a 1D batch
    # of PRNG `key` arguments. This means every image resulting from this batch will
    # be calculated using different randomization for the stochastic layers. The
    # optional axis name "batch" is provided as an assumed standard in ORGANIC when
    # layers need access to the batch dimension for parallel computations (e.g.
    # `equinox.nn.BatchNorm`).
    gen_img_batch, gen_state = jax.vmap(
        gen, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(z_in, gen_state, key=subkeys_gen_batch)

    # Compute critic estimates for batch. Note that the critic is assumed
    # to be in inference mode. The same comments regarding `jax.vmap` as above apply.
    y_est_batch, crit_state = jax.vmap(
        crit, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(gen_img_batch, crit_state, key=subkeys_crit_batch)

    # Compute the final scalar loss over the generator image batch.
    gen_loss = -jnp.mean(y_est_batch)

    # Return loss and updated states of generator and critic.
    return gen_loss, (gen_state, crit_state)


@ft.partial(
    jax.jit,
    static_argnames=("gen_static", "crit_static", "opt_gen", "size_in", "batch_size"),
)
def _wgan_gen_make_step(
    gen_params: WGANComponent,
    gen_static: WGANComponent,
    gen_state: eqx.nn.State,
    crit_params: WGANComponent,
    crit_static: WGANComponent,
    crit_state: eqx.nn.State,
    opt_gen: optax.GradientTransformation,
    opt_gen_state: optax.OptState,
    *,
    key: jax.Array,
    size_in: int,
    batch_size: int,
) -> tuple[WGANComponent, eqx.nn.State, eqx.nn.State, optax.OptState, jax.Array]:
    """Function that performs a training step for the generator during WGAN training by
    minimizing the corresponding loss function.

    **Arguments**

    - `gen_params`: The generator's trainable parameters. This should generally only
        include components that return `True` under `eqx.is_array`.
    - `gen_static`: The generator's non-trainable components. This should generally only
        include components that return `False` under `eqx.is_array`. Note that this
        argument is supposed to be set to training mode.
    - `gen_state`: The generator's current state. An updated generator state is returned
        after evaluating the batch.
    - `crit_params`: The critic's trainable parameters. This should generally only
        include components that return `True` under `eqx.is_array`.
    - `crit_static`: The critic's non-trainable components. This should generally only
        include components that return `False` under `eqx.is_array`. Note that this
        argument is supposed to be set to inference mode.
    - `crit_state`: The critic's state (i.e. current status for stateful layers). This
        is included since e.g. spectral normalization enforces Lipschitz-continuity
        in the critic. This state is returned again for API compatibility, but should
        not be meaningfully updated since the critic is assumed to be in inference mode
        (see warning below).
    - `opt_gen`: Optax optimizer used for the generator updates.
    - `opt_gen_state`: Current state of the generator's Optax optimizer.
    - `key`: JAX PRNG key used to generate a batch of 1D generator random latent input
        vectors. Also used to provide randomness to possible stochastic layers.
    - `size_in`: Size of the 1D generator input vectors.
    - `batch_size`: The size of batch to use.

    **Returns**

    The updated generator parameters in `gen_params`, the updated generator state in
    `gen_state`, the critic state in `crit_state` (should not have been updated,
    see warning below), the updated generator optimizer state in `opt_gen_state` and
    the calculated loss value in `gen_loss`.

    !!! warning

        Note that the critic should be set in inference mode before calling this
        function. Its state should not be updated. Since this function is meant to be
        used in a JAX JIT-ed update step of the generator, it implements no way of
        checking or enforcing this, and trusts the caller made sure of this beforehand.
    """
    # Generate a batch of random latent input vectors for the generator. The first index
    # of the batched vectors will run over the different vectors.
    key, subkey1, subkey2 = jr.split(key, 3)
    z_in = jr.normal(key=subkey1, shape=(batch_size, size_in))

    # Get loss estimate and corresponding gradients (which are also estimated gradients
    # relative to the Wasserstein distance; though note the loss value returned here is
    # not exactly the Wasserstein distance since the part that's dependent on the
    # training distribution and the critic's parameters has been dropped) for the
    # generator parameters.
    (gen_loss, (gen_state, crit_state)), gen_params_grads = jax.value_and_grad(
        _wgan_gen_loss,
        argnums=0,
        has_aux=True,
    )(
        gen_params,
        gen_static,
        gen_state,
        crit_params,
        crit_static,
        crit_state,
        z_in,
        key=subkey2,
    )

    # Calculate generator parameter updates and new optimizer state.
    gen_params_updates, opt_gen_state = opt_gen.update(
        gen_params_grads, opt_gen_state, gen_params
    )
    # Apply generator parameter updates.
    gen_params = eqx.apply_updates(gen_params, gen_params_updates)

    return gen_params, gen_state, crit_state, opt_gen_state, gen_loss


def _wgan_crit_loss(
    crit_params: WGANComponent,
    crit_static: WGANComponent,
    crit_state: eqx.nn.State,
    gen_params: WGANComponent,
    gen_static: WGANComponent,
    gen_state: eqx.nn.State,
    training_img_batch: jax.Array,
    z_in: jax.Array,
    *,
    key: jax.Array,
) -> tuple[jax.Array, tuple[eqx.nn.State, eqx.nn.State, jax.Array, jax.Array]]:
    """Get the critic's loss value on a batch of random 1D latent input vectors for
    the generator and a batch of random images from the training set.

    **Arguments**

    - `crit_params`: The critic's trainable parameters. This should generally only
        include components that return `True` under `eqx.is_array`.
    - `crit_static`: The critic's non-trainable components. This should generally only
        include components that return `False` under `eqx.is_array`.
    - `crit_state`: The critic's state. An updated critic state is returned after
        evaluating the batch.
    - `gen_params`: The generator's trainable parameters. This should generally only
        include components that return `True` under `eqx.is_array`.
    - `gen_static`: The generator's non-trainable components. This should generally only
        include components that return `False` under `eqx.is_array`.
    - `gen_state`: The generator's state. Note that the generator's state should not be
        updated in any way meaningful way, since the generator should be in inference
        mode (see the warning below). It is still returned for API compatibility.
    - `training_img_batch`: A batch of training data images. This should be a 4D JAX
        array following (Batch, Channel, Y, X) array indexing.
    - `z_in`: The batch of 1D generator random latent input vectors. The first index
        should run over the different batch elements, the second over the vectors
        themselves.
    - `key`: JAX PRNG key for calling stochastic layers.

    **Returns**

    Returns a tuple `(crit_loss, (crit_state, gen_state, score_training_imgs,
    score_gen_imgs))`, where `crit_loss` is the WGAN critic loss over the batch.
    The state of the critic and generator, and the mean scores of the training and
    generated images are also returned as auxiliary data. If trained to optimality,
    the returned loss value becomes an estimate for the Wasserstein distance between
    the generated images and the distribution of the training data.

    !!! warning

        Note that the generator should be set in inference mode before calling this
        function. Since this function is meant to be used in a JAX JIT-ed update step of
        the critic, it implements no way of checking or enforcing this, and trusts the
        caller made sure of this beforehand.
    """
    # Combine params and static into full models again.
    gen = eqx.combine(gen_params, gen_static)
    crit = eqx.combine(crit_params, crit_static)

    # Generate PRNG keys for possible stochastic layers in model calls.
    keys = jr.split(key, z_in.shape[0] + 1)
    key, subkeys_gen_batch = keys[0], keys[1:]
    keys = jr.split(key, z_in.shape[0] + 1)
    key, subkeys_crit_batch1 = keys[0], keys[1:]
    keys = jr.split(key, z_in.shape[0] + 1)
    key, subkeys_crit_batch2 = keys[0], keys[1:]

    # Generate batch of images from the generator. Note that JAX automatically considers
    # input keywords to be batched over axis 0. This means we need to provide a 1D batch
    # of PRNG `key` arguments. This means every image resulting from this batch will
    # be calculated using different randomization for the stochastic layers. The
    # optional axis name "batch" is provided as an assumed standard in ORGANIC when
    # layers need access to the batch dimension for parallel computations (e.g.
    # `equinox.nn.BatchNorm`). Note that the generator is assumed to be in inference
    # mode.
    gen_img_batch, gen_state = jax.vmap(
        gen, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(z_in, gen_state, key=subkeys_gen_batch)

    # Compute critic estimates for generated image batch. The same comments regarding
    # `jax.vmap` as above apply.
    y_est_gen_batch, crit_state = jax.vmap(
        crit, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(gen_img_batch, crit_state, key=subkeys_crit_batch1)

    # Compute critic estimates for the training image batch. The same comments regarding
    # `jax.vmap` as above apply.
    y_est_train_batch, crit_state = jax.vmap(
        crit, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(training_img_batch, crit_state, key=subkeys_crit_batch2)

    # Compute the final scalar loss over the generator and training image batches.
    score_training_imgs, score_gen_imgs = (
        jnp.mean(y_est_train_batch),
        jnp.mean(y_est_gen_batch),
    )
    crit_loss = -(score_training_imgs - score_gen_imgs)

    # Return loss and updated states of critic and generator. Note we also include
    # the individual mean score values on the training and generated images.
    return crit_loss, (crit_state, gen_state, score_training_imgs, score_gen_imgs)


@ft.partial(
    jax.jit,
    static_argnames=("crit_static", "gen_static", "opt_crit", "size_in", "batch_size"),
)
def _wgan_crit_make_step(
    crit_params: WGANComponent,
    crit_static: WGANComponent,
    crit_state: eqx.nn.State,
    gen_params: WGANComponent,
    gen_static: WGANComponent,
    gen_state: eqx.nn.State,
    opt_crit: optax.GradientTransformation,
    opt_crit_state: optax.OptState,
    *,
    key: jax.Array,
    size_in: int,
    batch_size: int,
    training_img_batch: jax.Array,
) -> tuple[
    WGANComponent,
    eqx.nn.State,
    eqx.nn.State,
    optax.OptState,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Function that performs a training step for the critic during WGAN training by
    minimizing the corresponding loss function.

    **Arguments**

    - `crit_params`: The critic's trainable parameters. This should generally only
        include components that return `True` under `eqx.is_array`.
    - `crit_static`: The critic's non-trainable components. This should generally only
        include components that return `False` under `eqx.is_array`.
    - `crit_state`: The critic's state. An updated critic state is returned after
        evaluating the batch.
    - `gen_params`: The generator's trainable parameters. This should generally only
        include components that return `True` under `eqx.is_array`.
    - `gen_static`: The generator's non-trainable components. This should generally only
        include components that return `False` under `eqx.is_array`.
    - `gen_state`: The generator's state. Note that the generator's state should not be
        updated in any way meaningful way, since the generator should be in inference
        mode (see the warning below). It is still returned for API compatibility.
    - `opt_crit`: Optax optimizer used for the critic updates.
    - `opt_crit_state`: Current state of the critic's Optax optimizer.
    - `key`: JAX PRNG key used to generate a batch of 1D generator random latent input
        vectors. Also used to provide randomness to possible stochastic layers.
    - `size_in`: Size of the 1D generator input vectors.
    - `batch_size`: The size of batch to use.
    - `training_img_batch`: A batch of training data images. This should be a 4D JAX
       array following (Batch, Channel, Y, X) array indexing. Note that we have to pass
       a pre-generated batch of training images since currently data augmentation
       with e.g. Albumentations is still performed on the host. This makes the API
       slightly different than e.g. `_wgan_gen_make_step()`.

    **Returns**

    The updated critic parameters in `crit_params`, the updated critic state in
    `crit_state`, the generator state in `gen_state` (should not have been updated,
    see warning below), the updated critic optimizer state in `opt_crit_state` and
    the calculated loss value (i.e. the WGAN Wasserstein distance estimate) in
    `crit_loss`. The individual mean critic score for training and generated images
    from the critic are returned in `score_training_imgs` and `score_gen_imgs`.

    !!! warning

        Note that the generator should be set in inference mode before calling this
        function. Its state should not be updated. Since this function is meant to be
        used in a JAX JIT-ed update step of the critic, it implements no way of
        checking or enforcing this, and trusts the caller made sure of this.
    """
    # Generate a batch of random latent input vectors for the generator. The first index
    # of the batched vectors will run over the different vectors.
    key, subkey1, subkey2 = jr.split(key, 3)
    z_in = jr.normal(key=subkey1, shape=(batch_size, size_in))

    # Get loss estimate (corresponding to the Wasserstein estimate if trained to
    # optimality) and corresponding gradients for the critic parameters. Updated
    # states and individual mean scores for training and generated images are
    # returned as auxiliary data.
    (
        (crit_loss, (crit_state, gen_state, score_training_imgs, score_gen_imgs)),
        crit_params_grads,
    ) = jax.value_and_grad(
        _wgan_crit_loss,
        argnums=0,
        has_aux=True,
    )(
        crit_params,
        crit_static,
        crit_state,
        gen_params,
        gen_static,
        gen_state,
        training_img_batch,
        z_in,
        key=subkey2,
    )

    # Calculate critic parameter updates and new optimizer state.
    crit_params_updates, opt_crit_state = opt_crit.update(
        crit_params_grads, opt_crit_state, crit_params
    )
    # Apply critic parameter updates.
    crit_params = eqx.apply_updates(crit_params, crit_params_updates)

    return (
        crit_params,
        crit_state,
        gen_state,
        opt_crit_state,
        crit_loss,
        score_training_imgs,
        score_gen_imgs,
    )
