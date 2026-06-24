from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class WindConfig:
    """Configuration for wind speed and direction."""
    speed: float = 5.0          # m/s
    direction: float = 45.0     # degrees (0 = North, 90 = East, 180 = South, 270 = West)

@dataclass
class TerrainConfig:
    """Configuration for terrain slope and elevation grid generation."""
    # Method can be "flat", "slope" (constant gradient), or "sinusoidal" / "random"
    elevation_type: str = "sinusoidal"
    max_elevation: float = 200.0  # meters
    slope_angle_deg: float = 10.0 # for constant slope gradient
    aspect_deg: float = 90.0      # aspect of the constant slope

@dataclass
class VegetationConfig:
    """Configuration for fuel/vegetation density grid generation."""
    # Method can be "uniform" or "noisy" (using perlin/random noise)
    density_type: str = "noisy"
    base_density: float = 0.7     # 0.0 to 1.0
    noise_scale: float = 0.1      # standard deviation of random variation

@dataclass
class SimulationConfig:
    """Overall wildfire simulator configurations."""
    rows: int = 100
    cols: int = 100
    dx: float = 10.0              # cell width in meters
    dy: float = 10.0              # cell height in meters
    dt: float = 2.0               # timestep duration in seconds
    
    # Physics scaling factors
    base_ros_scale: float = 0.05  # scales vegetation density to base ROS (m/s)
    wind_factor_coeff: float = 0.1 # c_W factor in ROS equation
    slope_factor_coeff: float = 0.3 # c_S factor in ROS equation
    
    # Fuel properties (Byram's Fireline Intensity)
    heat_combustion: float = 18000.0 # kJ/kg
    max_fuel_load: float = 2.5       # kg/m^2 max vegetation load
    
    # Wind and environmental conditions
    wind: WindConfig = field(default_factory=WindConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    vegetation: VegetationConfig = field(default_factory=VegetationConfig)
    
    # Initial states
    ignition_points: List[Tuple[int, int]] = field(default_factory=lambda: [(50, 50)])
