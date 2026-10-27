from .autoencoder import ConvAutoencoder
from .efficientad_lite import EfficientADLite
from .padim import PaDiM
from .patchcore import PatchCore
from .proposed_model import LightPatch
from .student_teacher import StudentTeacher

METHODS = {
    "autoencoder": ConvAutoencoder,
    "padim": PaDiM,
    "patchcore": PatchCore,
    "student_teacher": StudentTeacher,
    "efficientad_lite": EfficientADLite,
    "lightpatch": LightPatch,
}


def build_model(method: str, model_cfg: dict | None = None, pretrained: bool = True):
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {sorted(METHODS)}")
    return METHODS[method](**{**(model_cfg or {}), "pretrained": pretrained})
