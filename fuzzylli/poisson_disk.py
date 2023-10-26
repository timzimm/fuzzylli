import numpy as np


class PeriodicPoissonDisk:
    def __init__(self, w, h, d, minimum_distance, seed=42, k=30):
        self.extends = np.array([w, h, d])

        self.cell_size = minimum_distance / np.sqrt(3)
        self.grid_w = np.ceil(w / self.cell_size).astype(int)
        self.grid_h = np.ceil(h / self.cell_size).astype(int)
        self.grid_d = np.ceil(d / self.cell_size).astype(int)
        self.extends_int = np.array([self.grid_w, self.grid_h, self.grid_d])

        self.grid = -1 * np.ones((self.grid_d * self.grid_h * self.grid_w, 3))
        self.minimum_distance = minimum_distance
        self.k = k

        start = np.array([w, h, d]) * np.random.rand(1, 3)
        self.insert_point(start)
        self.active = [start]
        self.rng = np.random.default_rng(seed)

    def insert_point(self, x):
        cell_on_grid = np.floor(x / self.cell_size).astype(int)
        cell_idx = np.ravel_multi_index(cell_on_grid.T, self.extends_int)
        self.grid[cell_idx] = x

    def generate_in_periodic_shell_around(self, center):
        u = self.rng.uniform()
        r = np.cbrt(
            (u * (2 * self.minimum_distance) ** 3)
            + ((1 - u) * self.minimum_distance**3)
        )
        phi = 2 * np.pi * self.rng.uniform()
        theta = np.arccos(self.rng.uniform(-1, 1))
        pt = (
            np.array(
                [
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta),
                ]
            )
            + center
        )
        pt_periodic = pt % self.extends
        return pt_periodic

    def is_valid_point(self, point):
        pt_on_grid = np.floor(point / self.cell_size).astype(int)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    neighbor_cell_idx = np.ravel_multi_index(
                        ((pt_on_grid + np.array([dx, dy, dz])) % self.extends_int).T,
                        self.extends_int,
                    )
                    if np.sum(self.grid[neighbor_cell_idx]) > -2:
                        delta = np.abs(self.grid[neighbor_cell_idx] - point)
                        delta = np.where(
                            delta > 0.5 * self.extends, delta - self.extends, delta
                        )
                        if np.linalg.norm(delta) <= self.minimum_distance:
                            return False

        return True

    def sample(self, N):
        samples = []
        while len(self.active) > 0 and len(samples) < N:
            idx = self.rng.integers(0, len(self.active))
            src = self.active[idx]
            found = False
            for i in range(self.k):
                new_pt = self.generate_in_periodic_shell_around(src)
                if self.is_valid_point(new_pt):
                    self.insert_point(new_pt)
                    self.active.append(new_pt)
                    samples.append(new_pt)
                    found = True
            if not found:
                del self.active[idx]
        samples = np.asarray(samples)
        selection = self.rng.choice(
            samples.shape[0], min(len(samples), N), replace=False
        ).astype(int)
        return samples[selection].squeeze()
