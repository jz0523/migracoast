"""Distance from a point to the nearest coastline."""

import math

import numpy as np
from shapely import STRtree
from shapely.geometry import Point
from shapely.ops import nearest_points

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


class CoastlineIndex:
    """Spatial index over coastline geometries for nearest-distance queries."""

    def __init__(self, coast_gdf):
        self._geoms = list(coast_gdf.geometry)
        if not self._geoms:
            raise ValueError("Coastline index requires at least one geometry")
        self._tree = STRtree(self._geoms)

    def distance_km(self, lat: float, lon: float) -> float:
        """Kilometres from this point to the closest point on any coastline."""
        point = Point(lon, lat)
        nearest_index = int(self._tree.nearest(point))
        _, closest = nearest_points(point, self._geoms[nearest_index])
        return haversine_km(lat, lon, closest.y, closest.x)

    def distances_km(self, lats, lons) -> np.ndarray:
        """Vectorised convenience wrapper over `distance_km`."""
        return np.array([self.distance_km(a, o) for a, o in zip(lats, lons)])
