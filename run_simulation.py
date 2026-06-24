import argparse
import os
import sys
from src.config import SimulationConfig, WindConfig, TerrainConfig, VegetationConfig
from src.simulator import WildfireSimulator
from src.visualization import plot_frame, animate_simulation
from src.data_exporter import export_simulation_to_numpy, export_simulation_to_pytorch

def main():
    parser = argparse.ArgumentParser(description="Wildfire Prediction Research Project Simulator")
    parser.add_argument("--rows", type=int, default=100, help="Grid rows")
    parser.add_argument("--cols", type=int, default=100, help="Grid columns")
    parser.add_argument("--steps", type=int, default=100, help="Max simulation steps")
    parser.add_argument("--wind-speed", type=float, default=8.0, help="Wind speed in m/s")
    parser.add_argument("--wind-dir", type=float, default=60.0, help="Wind direction in degrees (0=N, 90=E)")
    parser.add_argument("--terrain-type", type=str, default="sinusoidal", choices=["flat", "slope", "sinusoidal"], help="Terrain elevation type")
    parser.add_argument("--veg-density", type=float, default=0.75, help="Base vegetation density (0.0 to 1.0)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save simulation outputs")
    parser.add_argument("--animate", action="store_true", default=True, help="Create and save a GIF animation")
    
    args = parser.parse_args()
    
    # 1. Setup configurations
    print("[*] Configuring wildfire simulation environment...")
    wind_config = WindConfig(speed=args.wind_speed, direction=args.wind_dir)
    terrain_config = TerrainConfig(elevation_type=args.terrain_type)
    veg_config = VegetationConfig(base_density=args.veg_density)
    
    config = SimulationConfig(
        rows=args.rows,
        cols=args.cols,
        wind=wind_config,
        terrain=terrain_config,
        vegetation=veg_config,
        ignition_points=[(args.rows // 2, args.cols // 2), (args.rows // 3, args.cols // 3)]
    )
    
    # 2. Initialize Simulator
    print("[*] Initializing Cellular Automata grid...")
    sim = WildfireSimulator(config)
    
    # 3. Simulation Loop
    print(f"[*] Running simulation up to {args.steps} steps...")
    active = True
    step_count = 0
    
    while active and step_count < args.steps:
        active = sim.step()
        step_count += 1
        if step_count % 10 == 0 or not active:
            burning_cells = sum(sim.state_grid.flatten() == 1)
            burned_cells = sum(sim.state_grid.flatten() == 2)
            print(f"    - Timestep {step_count:03d}: {burning_cells} cells burning, {burned_cells} cells burned.")
            
    print(f"[*] Simulation ended at timestep {step_count}. Total recorded timesteps: {len(sim.history)}.")
    
    # 4. Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 5. Export Data
    print("[*] Exporting simulation results for PyTorch training...")
    npz_path = os.path.join(args.output_dir, "simulation_data.npz")
    pt_path = os.path.join(args.output_dir, "simulation_tensor.pt")
    
    export_simulation_to_numpy(sim.history, npz_path)
    tensor = export_simulation_to_pytorch(sim.history, pt_path)
    
    print(f"    - Saved raw NumPy structures to: {npz_path}")
    print(f"    - Saved PyTorch tensor of shape {list(tensor.shape)} to: {pt_path}")
    
    # 6. Save visualizations
    print("[*] Generating visualization plots...")
    last_frame_path = os.path.join(args.output_dir, "last_timestep.png")
    plot_frame(sim.history[-1], show=False, save_path=last_frame_path)
    print(f"    - Saved final frame layout to: {last_frame_path}")
    
    if args.animate and len(sim.history) > 1:
        print("[*] Rendering simulation GIF animation (this might take a few moments)...")
        gif_path = os.path.join(args.output_dir, "wildfire_spread.gif")
        animate_simulation(sim.history, save_path=gif_path, fps=8)
        print(f"    - Saved animation to: {gif_path}")
        
    print("[+] Done! Wildfire simulation run completed successfully.")

if __name__ == "__main__":
    main()
