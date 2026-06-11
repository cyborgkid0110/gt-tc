"""Polygonal region geometry for coverage-deployment scenarios.

A Region is the deployable area for nodes: the square bounds, minus obstacle
polygons, optionally restricted to buffered path corridors. Built from a plain
dict (loaded from a YAML definition).
"""
import numpy as np
from shapely.geometry import Point, Polygon, LineString, box
from shapely.ops import nearest_points, unary_union


class Region:
    def __init__(self, area, deployable, bs_pos):
        self.area = area
        self.deployable = deployable          # shapely (multi)polygon
        self.bs_pos = (float(bs_pos[0]), float(bs_pos[1]))
        if not self.deployable.contains(Point(self.bs_pos)):
            raise ValueError(
                f"Base station {self.bs_pos} is not in the deployable area "
                f"(inside an obstacle or off the paths).")

    @classmethod
    def from_def(cls, d):
        area = float(d['area'])
        bounds = box(-area, -area, area, area)

        obstacles = [Polygon(ring) for ring in d.get('obstacles', [])]
        obstacles += [Point(float(cx), float(cy)).buffer(float(rad))
                      for cx, cy, rad in d.get('circles', [])]
        obstacle_union = unary_union(obstacles) if obstacles else None

        paths = d.get('paths', [])
        if paths:
            corridors = [LineString(p['coords']).buffer(float(p['width']) / 2.0)
                         for p in paths]
            deployable = unary_union(corridors).intersection(bounds)
        else:
            deployable = bounds

        if obstacle_union is not None:
            deployable = deployable.difference(obstacle_union)

        return cls(area, deployable, d['bs'])

    def contains(self, pt):
        return self.deployable.contains(Point(float(pt[0]), float(pt[1])))

    def sample_inside(self, n, rng):
        """Rejection-sample n points uniformly inside the deployable area."""
        minx, miny, maxx, maxy = self.deployable.bounds
        out = []
        while len(out) < n:
            xs = rng.uniform(minx, maxx, size=n)
            ys = rng.uniform(miny, maxy, size=n)
            for x, y in zip(xs, ys):
                if self.deployable.contains(Point(x, y)):
                    out.append((x, y))
                    if len(out) == n:
                        break
        return np.array(out, dtype=float)

    def project(self, pt):
        """Nearest point inside the deployable area (identity if already in)."""
        p = Point(float(pt[0]), float(pt[1]))
        if self.deployable.contains(p):
            return (p.x, p.y)
        nearest = nearest_points(self.deployable, p)[0]
        return (nearest.x, nearest.y)

    def coverage_targets(self, spacing):
        """A fixed grid of points inside the deployable area, for scoring."""
        minx, miny, maxx, maxy = self.deployable.bounds
        xs = np.arange(minx, maxx + spacing, spacing)
        ys = np.arange(miny, maxy + spacing, spacing)
        pts = [(x, y) for x in xs for y in ys
               if self.deployable.contains(Point(x, y))]
        return np.array(pts, dtype=float)
