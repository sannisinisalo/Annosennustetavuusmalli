# -*- coding: utf-8 -*-
"""
Luotu Ti 14.4.2026
Tekijä: Sanni Sinisalo

Luokkarakenne loppukäsittelyn koodeille.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Keskitetty juuripolku:
BASE_DIR = Path(r"C:\Users\User01\GRADU\Aineisto")


@dataclass(frozen=True)
class DoseMetricsConfig:
    organ_config: Optional[dict] = None
    dx_percentages: Optional[list] = None
    vx_thresholds: Optional[list] = None
    vx_organs: Optional[list] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "organ_config",
            self.organ_config
            or {
                "PTV": 1,
                "Heart": 2,
                "Contralateral lung": 3,
                "Ipsilateral lung": 4,
                "Contralateral breast": 5,
            },
        )
        object.__setattr__(
            self,
            "dx_percentages",
            self.dx_percentages or [98, 95, 90, 75, 50, 25, 10, 2],
        )
        object.__setattr__(self, "vx_thresholds", self.vx_thresholds or [35, 16, 8, 4])
        object.__setattr__(
            self,
            "vx_organs",
            self.vx_organs
            or [
                "PTV",
                "Heart",
                "Contralateral lung",
                "Ipsilateral lung",
                "Contralateral breast",
            ],
        )
