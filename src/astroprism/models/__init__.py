"""Models for Bayesian inference on multi-channel astronomical images."""

from astroprism.models.forward import ForwardModel
from astroprism.models.likelihood import LikelihoodModel, build_likelihood
from astroprism.models.noise import NoiseModel
from astroprism.models.response import InstrumentResponse
from astroprism.models.sky import DiffuseField, PointSourceField, SkyComponent, SkyModel

# Back-compat aliases (pre-rename names used by older notebooks/scripts).
FieldModel = DiffuseField
SignalModel = SkyComponent

__all__ = [
    "ForwardModel",
    "InstrumentResponse",
    "LikelihoodModel",
    "NoiseModel",
    "SkyComponent",
    "SkyModel",
    "DiffuseField",
    "PointSourceField",
    "build_likelihood",
    # back-compat
    "FieldModel",
    "SignalModel",
]
