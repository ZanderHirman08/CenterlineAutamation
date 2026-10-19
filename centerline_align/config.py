"""Settings for the Align tool. Distances are in the imagery's linear units
(US feet in Georgia State Plane)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassifierConfig:
    # Training labels come from the local road file itself (distances in CRS units).
    positive_half_width: float = 4.0  # pixels within this distance of a road = road
    negative_min_dist: float = 45.0   # pixels between these distances from any road = not road
    negative_max_dist: float = 120.0
    samples_per_class: int = 20000
    training_grids: int = 8           # how many grids to sample training pixels from


@dataclass
class RefineConfig:
    station_spacing: float = 10.0     # distance between cross-sections along a road
    search_half_width: float = 40.0   # how far each side of the line to look
    max_shift: float = 30.0           # never move a line farther than this
    min_road_width: float = 10.0      # plausible pavement widths
    max_road_width: float = 60.0
    simplify_tolerance: float = 0.5
    node_snap_tolerance: float = 1.0  # input endpoints closer than this are one node
    node_taper: float = 50.0          # distance over which an end's move into its node fades out
    review_min_confidence: float = 0.2
    review_max_shift: float = 15.0
    junction_mask: float = 30.0       # ignore cross-sections this close to an intersection
    offset_stiffness: float = 0.5     # cost of the line bending sideways (per ft^2 per station)
    width_stiffness: float = 0.25     # cost of the width changing between stations
    width_prior: float = 0.1          # pull toward the line's typical width (narrowing)
    stay_prior: float = 0.03          # pull toward the original position (no evidence = no move)
    # Flagged lines (REVIEW = 1) keep their original geometry instead of a doubtful move:
    # checked against hand-validated roads, moves under trees and big jumps made lines
    # worse more often than better.
    keep_flagged: bool = True
