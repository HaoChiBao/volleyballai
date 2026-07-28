"""
FIVB indoor volleyball metric ratios (canonical constraints).

Court (playing surface):
  length L = 18 m   (endline → endline)
  width  W =  9 m   (sideline → sideline)
  ratio  L/W = 2

Net (vertical plane on the center line):
  spans full court width W = 9 m between antennas
  sits on the mid-court line at L/2 = 9 m from each endline
  top height H = 2.24 m (women) or 2.43 m (men)
  mesh depth D ≈ 1.0 m (top tape → bottom tape)

Locked ratios used by the dual-outline solver:
  net_width / court_width     = 1
  court_length / net_width    = 2
  court_half / net_width      = 1   (each side is a 9×9 m square)
  attack_line offset from net = 3 m = net_width / 3
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FivbIndoor:
    length_m: float = 18.0
    width_m: float = 9.0
    net_height_m: float = 2.24  # women default
    net_depth_m: float = 1.0

    @property
    def mid_m(self) -> float:
        return self.length_m / 2.0

    @property
    def half_m(self) -> float:
        return self.length_m / 2.0

    @property
    def attack_offset_m(self) -> float:
        """Distance from center line to each attack line."""
        return 3.0 * (self.width_m / 9.0)

    @property
    def length_over_width(self) -> float:
        return self.length_m / self.width_m

    @property
    def length_over_net_width(self) -> float:
        # Playing net width == court width.
        return self.length_m / self.width_m

    @property
    def net_width_over_court_width(self) -> float:
        return 1.0

    def ratios(self) -> dict[str, float]:
        return {
            "court_length_m": self.length_m,
            "court_width_m": self.width_m,
            "net_width_m": self.width_m,
            "net_height_m": self.net_height_m,
            "net_depth_m": self.net_depth_m,
            "court_L_over_W": self.length_over_width,
            "court_L_over_net_W": self.length_over_net_width,
            "net_W_over_court_W": self.net_width_over_court_width,
            "half_court_m": self.half_m,
            "attack_from_net_m": self.attack_offset_m,
        }


DEFAULT_FIVB = FivbIndoor()
