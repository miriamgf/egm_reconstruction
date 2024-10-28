import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np


class Oclussion:

    def __init__(self, bsps_64, patches_oclussion):

        self.bsps_64 = bsps_64
        self.patches_off = [patches_oclussion]

    def turn_off_patches(self):

        patch_indices = {
            "P1": (
                slice(None),
                slice(0, 3),
                slice(0, 4),
            ),  # B6-A4: filas 0-2, columnas 0-3 #TOP FRONT LEFT
            "P2": (
                slice(None),
                slice(3, 6),
                slice(0, 4),
            ),  # B3-A1: filas 3-5, columnas 0-3 #BOTTOM FRONT RIGHT
            "P3": (
                slice(None),
                slice(0, 3),
                slice(6, 10),
            ),  # D12-C4 : filas 0-2, columnas 6-8 #TOP BACK
            "P4": (
                slice(None),
                slice(3, 6),
                slice(6, 10),
            ),  # D9-C1 : filas 3-5, columnas 6-8 #BOTTOM BACK
            "P5": (
                slice(None),
                slice(0, 3),
                slice(12, 16),
            ),  # B6-A4: filas 0-2, columnas 8-11 #TOP FRONT RIGHT (Repetido)
            "P6": (
                slice(None),
                slice(3, 6),
                slice(12, 16),
            ),  # B3-A1: filas 3-5, columnas 8-11 #BOTTOM FRONT RIGHT (Repetido)
            "P7": (slice(None), slice(0, 3), slice(4, 6)),  # ILR4-R7: TOP RIGHT SIDE
            "P8": (slice(None), slice(3, 6), slice(4, 6)),  # ILR4-R7: BOTTOM RIGHT SIDE
            "P9": (slice(None), slice(0, 3), slice(10, 12)),  # ILR4-R7: TOP LEFT SIDE
            "P10": (
                slice(None),
                slice(3, 6),
                slice(10, 12),
            ),  # ILR4-R7: BOTTOM LEFT SIDE
        }

        new_bsps = self.bsps_64

        for patch in self.patches_off:
            if patch in self.patches_off:
                new_bsps[patch_indices[patch]] = self.bsps_64[
                    patch_indices[patch]
                ].mean()  # If patches are 0 it creates NaNs in the network

        return new_bsps

    def get_patches_name(self, bsps_64):
        """
        Get names of patches in bsps_64

        Parameters:
            bsps_64:
        Return:
            patches: dictionary with patche name as key and bsps as value.
        """
        patches = {}

        index = 1
        for i in range(0, 12):
            patches["A{0}".format(index)] = bsps_64[i]
            index += 1

        index = 1
        for i in range(12, 24):
            patches["B{0}".format(index)] = bsps_64[i]
            index += 1

        index = 1
        for i in range(24, 36):
            patches["C{0}".format(index)] = bsps_64[i]
            index += 1

        index = 1
        for i in range(36, 48):
            patches["D{0}".format(index)] = bsps_64[i]
            index += 1

        index = 1
        for i in range(48, 56):
            patches["L{0}".format(index)] = bsps_64[i]
            index += 1

        index = 1
        for i in range(56, 64):
            patches["R{0}".format(index)] = bsps_64[i]
            index += 1

        return patches