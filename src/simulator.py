import numpy as np
from typing import List, Dict, Any, Tuple
from src.config import SimulationConfig
from src.physics import compute_slope_and_aspect, calculate_directional_ros, compute_fireline_intensity

class WildfireSimulator:
    """
    Cellular Automata Wildfire Simulator.
    Manages a 2D grid of states:
      0 = Unburned
      1 = Burning
      2 = Burned
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rows = config.rows
        self.cols = config.cols
        
        # Initialize grids
        self.state_grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.elevation_grid = self._generate_elevation()
        self.vegetation_density = self._generate_vegetation()
        
        # Fuel tracking (dynamic during simulation)
        self.fuel_grid = self.vegetation_density.copy()
        
        # Precompute slope and aspect
        self.slope_grid, self.aspect_grid = compute_slope_and_aspect(
            self.elevation_grid, self.config.dx, self.config.dy
        )
        
        # Precompute Potential ROS map for all cells (independent of active fire)
        self.potential_ros_map = self._compute_potential_ros()
        
        # Current active ROS and intensity maps
        self.active_ros_map = np.zeros((self.rows, self.cols), dtype=float)
        self.intensity_map = np.zeros((self.rows, self.cols), dtype=float)
        
        # Ignite initial points
        for r, c in self.config.ignition_points:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.state_grid[r, c] = 1 # Burning
                
        # History storage
        self.history: List[Dict[str, Any]] = []
        self.timestep = 0
        self._record_timestep()

    def _generate_elevation(self) -> np.ndarray:
        """Generate elevation grid based on configuration."""
        y = np.linspace(0, self.rows * self.config.dy, self.rows)
        x = np.linspace(0, self.cols * self.config.dx, self.cols)
        X, Y = np.meshgrid(x, y)
        
        if self.config.terrain.elevation_type == "flat":
            return np.zeros((self.rows, self.cols))
        elif self.config.terrain.elevation_type == "slope":
            # Constant slope
            angle_rad = np.radians(self.config.terrain.slope_angle_deg)
            aspect_rad = np.radians(self.config.terrain.aspect_deg)
            # Elevation increases in aspect direction
            # aspect 0 is North (-y), aspect 90 is East (+x)
            Z = np.tan(angle_rad) * (X * np.sin(aspect_rad) - Y * np.cos(aspect_rad))
            # Shift Z to be positive
            Z -= Z.min()
            return Z
        else:
            # Sinusoidal / wavy terrain (default)
            Z = (self.config.terrain.max_elevation / 2.0) * (
                np.sin(X / 200.0) * np.cos(Y / 200.0) + 1.0
            )
            return Z

    def _generate_vegetation(self) -> np.ndarray:
        """Generate vegetation density grid."""
        if self.config.vegetation.density_type == "uniform":
            return np.full((self.rows, self.cols), self.config.vegetation.base_density)
        else:
            # Generate a noisy pattern using random numbers smoothed with a Gaussian-like filter
            raw_noise = np.random.randn(self.rows, self.cols)
            # Smooth using simple 2D convolution pad and average
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            smoothed = np.zeros_like(raw_noise)
            # Apply padding
            padded = np.pad(raw_noise, kernel_size // 2, mode='edge')
            for r in range(self.rows):
                for c in range(self.cols):
                    smoothed[r, c] = np.sum(padded[r:r+kernel_size, c:c+kernel_size] * kernel)
            
            # Scale and clip
            veg = self.config.vegetation.base_density + smoothed * self.config.vegetation.noise_scale
            return np.clip(veg, 0.0, 1.0)

    def _compute_potential_ros(self) -> np.ndarray:
        """
        Compute the Potential Rate of Spread (ROS) for every cell.
        This represents the ROS if the fire were propagating in the optimal direction (aligned with wind and slope).
        """
        r0 = self.config.base_ros_scale * self.vegetation_density
        # Max wind factor (when aligned)
        phi_w = self.config.wind_factor_coeff * self.config.wind.speed
        # Max slope factor (when aligned uphill)
        phi_s = self.config.slope_factor_coeff * np.tan(self.slope_grid)
        
        return r0 * (1.0 + phi_w + phi_s)

    def _record_timestep(self) -> None:
        """Record the current state of the simulation to history."""
        self.history.append({
            "timestep": self.timestep,
            "state_grid": self.state_grid.copy(),
            "potential_ros_map": self.potential_ros_map.copy(),
            "active_ros_map": self.active_ros_map.copy(),
            "intensity_map": self.intensity_map.copy(),
            "wind_speed": self.config.wind.speed,
            "wind_direction": self.config.wind.direction,
            "slope": self.slope_grid.copy(),
            "aspect": self.aspect_grid.copy(),
            "vegetation_density": self.vegetation_density.copy(),
            "fuel_grid": self.fuel_grid.copy()
        })

    def step(self) -> bool:
        """
        Execute one timestep of the Cellular Automata.
        Returns:
            bool: True if there was a change in state or fire is still active, False if fire is extinguished.
        """
        # Find currently burning cells
        burning_indices = np.argwhere(self.state_grid == 1)
        if len(burning_indices) == 0:
            return False
            
        next_state = self.state_grid.copy()
        next_fuel = self.fuel_grid.copy()
        
        # Track ignition probabilities for each cell
        # Using dictionary to accumulate probabilities: (r, c) -> list of 1 - P(i -> j)
        ignition_factors: Dict[Tuple[int, int], List[float]] = {}
        
        # Reset active ROS map for this timestep
        self.active_ros_map.fill(0.0)
        
        # 8-connected neighbors (Moore neighborhood)
        neighbors_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for r, c in burning_indices:
            # 1. Consume fuel in the burning cell
            # Fuel is consumed proportional to vegetation density and dt
            # Let's say it takes 20 seconds of burning to consume fully
            fuel_consumed = (1.0 / 20.0) * self.config.dt
            next_fuel[r, c] = max(0.0, self.fuel_grid[r, c] - fuel_consumed)
            
            # If fuel is completely depleted, cell is burned out
            if next_fuel[r, c] <= 0.0:
                next_state[r, c] = 2 # Burned
                
            # 2. Spread fire to neighbors
            for dr, dc in neighbors_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.state_grid[nr, nc] == 0: # Only spread to unburned cells
                        # Calculate Rate of Spread from (r, c) to (nr, nc)
                        ros = calculate_directional_ros(
                            (r, c), (nr, nc), self.config,
                            self.vegetation_density, self.slope_grid, self.aspect_grid
                        )
                        
                        # Distance scaling factor
                        dist = np.sqrt(dr**2 + dc**2)
                        
                        # Probability of ignition
                        # P = 1 - exp(-ros * dt / dist)
                        p_ignite = 1.0 - np.exp(- (ros * self.config.dt) / dist)
                        
                        if p_ignite > 0:
                            # Update the active ROS map at the burning cell
                            self.active_ros_map[r, c] = max(self.active_ros_map[r, c], ros)
                            
                            # Keep track of survival factor: (1 - P)
                            if (nr, nc) not in ignition_factors:
                                ignition_factors[(nr, nc)] = []
                            ignition_factors[(nr, nc)].append(1.0 - p_ignite)
                            
        # 3. Apply probabilistic ignitions
        for (r, c), survival_list in ignition_factors.items():
            # Overall P_ignite = 1 - product(survival_list)
            p_total = 1.0 - np.prod(survival_list)
            if np.random.rand() < p_total:
                next_state[r, c] = 1 # Ignite!
                
        # Update state and fuel grids
        self.state_grid = next_state
        self.fuel_grid = next_fuel
        
        # Calculate fireline intensity based on current active ROS and state
        self.intensity_map = compute_fireline_intensity(
            self.active_ros_map, self.vegetation_density, self.state_grid, self.config
        )
        
        self.timestep += 1
        self._record_timestep()
        
        # Return True if any cell is still burning
        return np.any(self.state_grid == 1)
