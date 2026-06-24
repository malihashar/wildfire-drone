import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Any

def plot_frame(step_data: Dict[str, Any], show: bool = True, save_path: str = None) -> None:
    """
    Plot a single timestep of the simulation in a 3-panel subplot layout:
      - Left: Fire State Grid
      - Middle: Potential ROS / Active ROS Map
      - Right: Fireline Intensity Map
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Define custom colormap for fire states
    # 0 = Unburned (forest green), 1 = Burning (bright orange-red), 2 = Burned (charcoal gray)
    state_cmap = mcolors.ListedColormap(['#2E7D32', '#FF3D00', '#424242'])
    state_bounds = [-0.5, 0.5, 1.5, 2.5]
    state_norm = mcolors.BoundaryNorm(state_bounds, state_cmap.N)
    
    # 1. State grid plot
    im0 = axes[0].imshow(step_data["state_grid"], cmap=state_cmap, norm=state_norm)
    axes[0].set_title(f"Wildfire State (Timestep: {step_data['timestep']})", fontsize=12, fontweight='bold')
    cbar0 = fig.colorbar(im0, ax=axes[0], ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar0.ax.set_yticklabels(['Unburned', 'Burning', 'Burned'])
    
    # 2. Rate of Spread plot (Potential ROS is more complete; active is where fire is active)
    ros_data = step_data["potential_ros_map"]
    im1 = axes[1].imshow(ros_data, cmap='inferno')
    axes[1].set_title("Potential Rate of Spread (m/s)", fontsize=12, fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Fireline Intensity plot
    intensity_data = step_data["intensity_map"]
    im2 = axes[2].imshow(intensity_data, cmap='hot')
    axes[2].set_title("Fireline Intensity (kW/m)", fontsize=12, fontweight='bold')
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Draw wind direction indicator on the first plot if wind is non-zero
    wind_speed = step_data.get("wind_speed", 0.0)
    wind_dir = step_data.get("wind_direction", 0.0)
    if wind_speed > 0:
        # Map wind angle (0=N, 90=E, etc.) to direction vector
        # Angle 0 is pointing along -y (upward in imshow)
        angle_rad = np.radians(wind_dir)
        dy = -np.cos(angle_rad)
        dx = np.sin(angle_rad)
        
        # Position indicator in bottom-left corner of the state plot
        rows, cols = step_data["state_grid"].shape
        axes[0].arrow(
            cols * 0.08, rows * 0.92, dx * (cols * 0.05), dy * (rows * 0.05),
            head_width=cols*0.02, head_length=rows*0.02, fc='white', ec='white', width=cols*0.003
        )
        axes[0].text(
            cols * 0.08, rows * 0.98, f"Wind: {wind_speed}m/s",
            color='white', fontsize=9, fontweight='bold', ha='left'
        )
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def animate_simulation(history: List[Dict[str, Any]], save_path: str = None, fps: int = 5) -> FuncAnimation:
    """
    Generate a matplotlib animation of the full wildfire simulation history.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    state_cmap = mcolors.ListedColormap(['#2E7D32', '#FF3D00', '#424242'])
    state_bounds = [-0.5, 0.5, 1.5, 2.5]
    state_norm = mcolors.BoundaryNorm(state_bounds, state_cmap.N)
    
    # Initial frame
    step_data = history[0]
    im0 = axes[0].imshow(step_data["state_grid"], cmap=state_cmap, norm=state_norm)
    cbar0 = fig.colorbar(im0, ax=axes[0], ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar0.ax.set_yticklabels(['Unburned', 'Burning', 'Burned'])
    
    # Max limits for scaling colorbars
    max_ros = max([step["potential_ros_map"].max() for step in history] + [1e-3])
    max_intensity = max([step["intensity_map"].max() for step in history] + [1e-3])
    
    im1 = axes[1].imshow(step_data["potential_ros_map"], cmap='inferno', vmin=0, vmax=max_ros)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(step_data["intensity_map"], cmap='hot', vmin=0, vmax=max_intensity)
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Set titles
    axes[0].set_title(f"Wildfire State (Timestep: 0)", fontsize=12, fontweight='bold')
    axes[1].set_title("Potential Rate of Spread (m/s)", fontsize=12, fontweight='bold')
    axes[2].set_title("Fireline Intensity (kW/m)", fontsize=12, fontweight='bold')
    
    # Wind indicator setup
    wind_arrow = None
    wind_text = None
    
    def update(frame: int):
        nonlocal wind_arrow, wind_text
        step = history[frame]
        
        # Update image data
        im0.set_array(step["state_grid"])
        im1.set_array(step["potential_ros_map"])
        im2.set_array(step["intensity_map"])
        
        # Update title
        axes[0].set_title(f"Wildfire State (Timestep: {step['timestep']})", fontsize=12, fontweight='bold')
        
        # Redraw wind arrow
        if wind_arrow:
            wind_arrow.remove()
        if wind_text:
            wind_text.remove()
            
        wind_speed = step.get("wind_speed", 0.0)
        wind_dir = step.get("wind_direction", 0.0)
        if wind_speed > 0:
            angle_rad = np.radians(wind_dir)
            dy = -np.cos(angle_rad)
            dx = np.sin(angle_rad)
            rows, cols = step["state_grid"].shape
            wind_arrow = axes[0].arrow(
                cols * 0.08, rows * 0.92, dx * (cols * 0.05), dy * (rows * 0.05),
                head_width=cols*0.02, head_length=rows*0.02, fc='white', ec='white', width=cols*0.003
            )
            wind_text = axes[0].text(
                cols * 0.08, rows * 0.98, f"Wind: {wind_speed}m/s",
                color='white', fontsize=9, fontweight='bold', ha='left'
            )
            
        return im0, im1, im2
        
    plt.tight_layout()
    
    anim = FuncAnimation(fig, update, frames=len(history), interval=1000 // fps, blit=False)
    
    if save_path:
        # Save as video or gif
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps)
            
    plt.close()
    return anim
