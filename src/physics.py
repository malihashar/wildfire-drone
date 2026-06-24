import numpy as np
from typing import Tuple
from src.config import SimulationConfig

def compute_terrain_derivatives(elevation: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial derivatives of elevation using Sobel filters or central differences.
    Returns:
        dZ_dy: change in elevation along the row axis (y-axis, pointing South)
        dZ_dx: change in elevation along the column axis (x-axis, pointing East)
    """
    # Use central differences with border padding to keep dimensions matching
    # dZ/dy: (Z[r+1, c] - Z[r-1, c]) / (2 * dy)
    dZ_dy = np.zeros_like(elevation)
    dZ_dx = np.zeros_like(elevation)
    
    # Internal cells
    dZ_dy[1:-1, :] = (elevation[2:, :] - elevation[:-2, :]) / (2.0 * dy)
    dZ_dx[:, 1:-1] = (elevation[:, 2:] - elevation[:, :-2]) / (2.0 * dx)
    
    # Boundary cells: forward/backward differences
    dZ_dy[0, :] = (elevation[1, :] - elevation[0, :]) / dy
    dZ_dy[-1, :] = (elevation[-1, :] - elevation[-2, :]) / dy
    
    dZ_dx[:, 0] = (elevation[:, 1] - elevation[:, 0]) / dx
    dZ_dx[:, -1] = (elevation[:, -1] - elevation[:, -2]) / dx
    
    return dZ_dy, dZ_dx

def compute_slope_and_aspect(elevation: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the slope angle (in radians) and the aspect angle (in degrees).
    Aspect matches wind direction orientation: 0 degrees is North, 90 is East, etc.
    pointing in the UPHILL direction.
    """
    dZ_dy, dZ_dx = compute_terrain_derivatives(elevation, dx, dy)
    
    # slope = arctan(sqrt((dZ/dx)^2 + (dZ/dy)^2))
    grad_magnitude = np.sqrt(dZ_dx**2 + dZ_dy**2)
    slope = np.arctan(grad_magnitude) # in radians
    
    # Aspect: direction of steepest ascent (uphill)
    # y increases downwards, so North is -y.
    # atan2(dx, -dy) gives angle relative to -y axis.
    aspect = np.degrees(np.arctan2(dZ_dx, -dZ_dy)) % 360.0
    
    return slope, aspect

def calculate_directional_ros(
    from_coords: Tuple[int, int],
    to_coords: Tuple[int, int],
    config: SimulationConfig,
    vegetation_density: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray
) -> float:
    """
    Calculate the Rate of Spread (ROS) from one cell to an adjacent cell.
    
    Equations:
    ROS = R_0 * (1 + Phi_W + Phi_S)
    R_0 = base_ros_scale * vegetation_density
    Phi_W = wind_factor_coeff * wind_speed * cos(spread_dir - wind_dir)
    Phi_S = slope_factor_coeff * tan(slope) * cos(spread_dir - uphill_dir)
    """
    r_from, c_from = from_coords
    r_to, c_to = to_coords
    
    # Target cell vegetation density
    veg = vegetation_density[r_to, c_to]
    if veg <= 0.0:
        return 0.0
        
    # Base ROS
    r0 = config.base_ros_scale * veg
    
    # Spread direction vector and angle
    dy = r_to - r_from
    dx = c_to - c_from
    
    # Distance between cells
    dist = np.sqrt((dy * config.dy)**2 + (dx * config.dx)**2)
    
    # Propagation direction angle in degrees
    theta_d = np.degrees(np.arctan2(dx, -dy)) % 360.0
    
    # 1. Wind component
    wind_dir_rad = np.radians(config.wind.direction)
    theta_d_rad = np.radians(theta_d)
    cos_wind = np.cos(theta_d_rad - wind_dir_rad)
    # Wind only aids spread if it is blowing in a similar direction (cos > 0)
    phi_w = config.wind_factor_coeff * config.wind.speed * max(0.0, cos_wind)
    
    # 2. Slope component
    target_slope = slope[r_to, c_to]       # in radians
    target_aspect = aspect[r_to, c_to]     # in degrees (uphill direction)
    cos_slope = np.cos(theta_d_rad - np.radians(target_aspect))
    # Slope only aids spread uphill (cos > 0)
    phi_s = config.slope_factor_coeff * np.tan(target_slope) * max(0.0, cos_slope)
    
    ros = r0 * (1.0 + phi_w + phi_s)
    return float(max(0.0, ros))

def compute_fireline_intensity(
    ros_map: np.ndarray,
    vegetation_density: np.ndarray,
    state_grid: np.ndarray,
    config: SimulationConfig
) -> np.ndarray:
    """
    Calculate Fireline Intensity using Byram's formula: I = H * w * r
    Where:
        I = fireline intensity (kW/m)
        H = low heat of combustion (kJ/kg)
        w = fuel loaded per unit area (kg/m^2) = max_fuel_load * vegetation_density
        r = Rate of Spread (m/s)
    Intensity is 0 for cells that are not currently burning.
    """
    w = config.max_fuel_load * vegetation_density
    intensity = config.heat_combustion * w * ros_map
    
    # Intensity is only non-zero for currently burning cells (state == 1)
    burning_mask = (state_grid == 1).astype(float)
    return intensity * burning_mask
