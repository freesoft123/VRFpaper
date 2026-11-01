# -*- coding: utf-8 -*-
"""20 x 10 x 10 cuboid geometry and square-opening definition for Appendix B.

The square opening is centered on the x=20 face. The simulation driver sets
the opening side length to 3, 5, 8, or 10 m.
"""
import numpy as np

import hex_frustum_geometry as H

Lx, Ly, Lz = 20.0, 10.0, 10.0


def build_cuboid():
    centre = np.array([Lx / 2, Ly / 2, Lz / 2])
    V = [(0, 0, 0), (Lx, 0, 0), (Lx, Ly, 0), (0, Ly, 0),
         (0, 0, Lz), (Lx, 0, Lz), (Lx, Ly, Lz), (0, Ly, Lz)]
    quads = [(0, 3, 2, 1),      # z=0
             (4, 5, 6, 7),      # z=Lz
             (0, 1, 5, 4),      # y=0
             (3, 7, 6, 2),      # y=Ly
             (0, 4, 7, 3),      # x=0
             (1, 2, 6, 5)]      # x=Lx (opening face)
    return [H.Face([V[i] for i in q], centre) for q in quads], 5
