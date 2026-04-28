"""Module for performing image reconstruction on OIFITS data."""

from typing import Literal

import organic.sparco as sparco


# TODO: implement once regular WGAN training works. Start with the monochromatic case.
# We can work on polychromatic later.
def img_rec(
    gen,  # TODO: add type hints
    crit,
    oidata,
    *,
    observables: Literal["VIS2", "T3PHI"],
    mu: float,
    sparco: sparco.Sparco | None,
    ft_mode: Literal["FFT"],
    fft_padding: tuple[int, int] | None,
    diagnostics: bool = False,
    nboot: int | None = None,
) -> None:
    """Perform image reconstruction on OI data given a pre-trained WGAN generator
    and critic.

    **Arguments**

    - `gen`: The WGAN image generator.
    - `crit`: The WGAN critic.
    - `oidata`: OIFITS data object containing observational data.
    - `observables`: Tuple of strings with which observables to consider in the data
        term of the loss function (e.g. `("VIS2", "T3PHI")` to consider squared
        visibilities and closure phases).
    - `mu`: Regularization weight in the total loss function for the Wasserstein loss.
    - `sparco`: Specification of the SPARCO method if representing the central objects
        in a parametrized analytical way.
    - `ft_mode`: Manner in which to calculate the Fourier transform of the image under
        reconstruction.
    - `fft_padding`: Number of ($x$, $y$)-pixels to pad the image to before
        calculating the Fourier transform in case `ft_mode = "FFT"`.
    - `diagnostics`: Whether to provide diagnostic plots of the reconstruction process.
    - `nboot`: Number of bootstrap iterations to perform to assess the final image
        and its uncertainties.
    """
    raise NotImplementedError
