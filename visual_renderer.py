import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Any, Optional

import time

class MoonLanderVisualizer:
    """Simple 2D visualization for moon lander training"""
    
    def __init__(self, env, figsize=(12, 8), trail_length=50):
        self.env = env
        self.figsize = figsize
        self.trail_length = trail_length
        
        # Initialize plot
        self.fig, (self.ax_main, self.ax_info) = plt.subplots(1, 2, figsize=figsize, 
                                                               gridspec_kw={'width_ratios': [3, 1]})
        
        # State tracking
        self.trajectory = []
        self.thrust_history = []
        self.episode_data = []
        
        # Visual elements
        self.lander_patch = None
        self.thrust_line = None
        self.trail_line = None
        self.target_patch = None
        
        self.setup_plot()
    
    def setup_plot(self):
        """Setup the main plotting area"""
        # Main view setup
        self.ax_main.set_xlim(-150, 150)
        self.ax_main.set_ylim(-10, 120)
        self.ax_main.set_xlabel('X Position (m)')
        self.ax_main.set_ylabel('Z Position (m)')
        self.ax_main.set_title('Moon Lander Training Visualization')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect('equal')
        
        # Draw ground
        self.ax_main.axhline(y=0, color='gray', linewidth=3, label='Lunar Surface')
        
        # Draw target landing zone
        target_x, target_y, target_z = self.env.target_position
        self.target_patch = patches.Circle((target_x, target_z), 5, 
                                          fill=True, color='green', alpha=0.3, 
                                          label='Target Zone')
        self.ax_main.add_patch(self.target_patch)
        
        # Initialize lander representation (triangle pointing up)
        self.lander_patch = patches.RegularPolygon((0, 0), 3, radius=3, 
                                                  orientation=0, fill=True, 
                                                  color='blue', alpha=0.8)
        self.ax_main.add_patch(self.lander_patch)
        
        # Initialize thrust visualization
        self.thrust_line, = self.ax_main.plot([], [], 'r-', linewidth=3, alpha=0.8)
        
        # Initialize trajectory trail
        self.trail_line, = self.ax_main.plot([], [], 'b-', alpha=0.5, linewidth=1)
        
        # Info panel setup
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        self.ax_info.set_title('Flight Data')
        
        # Legend
        self.ax_main.legend(loc='upper right')
        
        plt.tight_layout()
    
    def update_display(self, state_info: Dict[str, Any], action: np.ndarray, 
                      step_count: int, episode_reward: float):
        """Update the visualization with current state"""
        pos = state_info['position']
        vel = state_info['velocity']
        
        # Store trajectory point
        self.trajectory.append([pos[0], pos[2]])  # x, z coordinates
        if len(self.trajectory) > self.trail_length:
            self.trajectory.pop(0)
        
        # Update lander position
        self.lander_patch.remove()
        self.lander_patch = patches.RegularPolygon((pos[0], pos[2]), 3, radius=3, 
                                                orientation=0, fill=True, 
                                                color='blue', alpha=0.8)
        self.ax_main.add_patch(self.lander_patch)
        
        # Update trajectory trail
        if len(self.trajectory) > 1:
            trail_array = np.array(self.trajectory)
            self.trail_line.set_data(trail_array[:, 0], trail_array[:, 1])
        
        # Update thrust visualization
        thrust_magnitude = action[0]
        gimbal_x, gimbal_y = action[1], action[2]
        
        if thrust_magnitude > 0.01:  # Only show thrust if significant
            # Calculate thrust direction (simplified 2D projection)
            thrust_length = thrust_magnitude * 10  # Scale for visibility
            thrust_angle = gimbal_x * np.pi/12  # Convert gimbal to angle
            
            thrust_end_x = pos[0] - thrust_length * np.sin(thrust_angle)
            thrust_end_z = pos[2] - thrust_length * np.cos(thrust_angle)
            
            self.thrust_line.set_data([pos[0], thrust_end_x], [pos[2], thrust_end_z])
        else:
            self.thrust_line.set_data([], [])
        
        # Update info panel
        self.ax_info.clear()
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        
        info_text = f"""Step: {step_count}
Reward: {episode_reward:.1f}

Position:
  X: {pos[0]:.1f} m
  Y: {pos[1]:.1f} m  
  Z: {pos[2]:.1f} m

Velocity:
  X: {vel[0]:.1f} m/s
  Y: {vel[1]:.1f} m/s
  Z: {vel[2]:.1f} m/s

Action:
  Thrust: {thrust_magnitude:.3f}
  Gimbal X: {gimbal_x:.3f}
  Gimbal Y: {gimbal_y:.3f}

Fuel: {state_info['fuel_ratio']:.1%}
Distance: {state_info['distance_to_target']:.1f} m"""
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    def reset_episode(self):
        """Reset visualization for new episode"""
        self.trajectory.clear()
        self.thrust_history.clear()
        self.trail_line.set_data([], [])
        self.thrust_line.set_data([], [])
    
    def show_episode_result(self, result: str, final_state: Dict[str, Any]):
        """Show episode completion result"""
        # Change lander color based on result
        if result == 'SUCCESS':
            self.lander_patch.set_color('green')
        elif result == 'CRASH':
            self.lander_patch.set_color('red')
        else:
            self.lander_patch.set_color('orange')
        
        # Add result text
        self.ax_main.text(0.02, 0.98, f"Result: {result}", 
                         transform=self.ax_main.transAxes,
                         fontsize=14, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))


class VisualTrainingRunner:
    """Run training with live visualization"""
    
    def __init__(self, env, agent, visualizer: MoonLanderVisualizer):
        self.env = env
        self.agent = agent
        self.visualizer = visualizer
        self.episode_count = 0
    
    def run_episode_with_visualization(self, delay: float = 0.05, max_steps: int = 1000):
        """Run a single episode with live visualization"""
        state = self.env.reset()
        self.visualizer.reset_episode()
        
        episode_reward = 0
        step_count = 0
        
        # Reset lander color
        self.visualizer.lander_patch.set_color('blue')
        
        print(f"\n=== Episode {self.episode_count} ===")
        
        for step in range(max_steps):
            # Get action from agent
            action, _, _ = self.agent.select_action(state, deterministic=True)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Update visualization
            state_info = self.env.get_state_info()
            self.visualizer.update_display(state_info, action, step_count, episode_reward)
            
            # Refresh display
            plt.pause(delay)
            
            state = next_state
            
            if done:
                # Show final result
                self.visualizer.show_episode_result(info.get('landing_result', 'Unknown'), state_info)
                plt.pause(1.0)  # Pause to see result
                
                print(f"Episode completed in {step_count} steps")
                print(f"Result: {info.get('landing_result', 'Unknown')}")
                print(f"Total reward: {episode_reward:.1f}")
                print(f"Final position: ({state_info['position'][0]:.1f}, {state_info['position'][1]:.1f}, {state_info['position'][2]:.1f})")
                print(f"Final velocity: ({state_info['velocity'][0]:.1f}, {state_info['velocity'][1]:.1f}, {state_info['velocity'][2]:.1f})")
                
                break
        
        self.episode_count += 1
        return episode_reward, step_count, info.get('landing_result', 'Unknown')
    
    def run_multiple_episodes(self, num_episodes: int = 5, delay: float = 0.05):
        """Run multiple episodes with visualization"""
        results = []
        
        for i in range(num_episodes):
            reward, steps, result = self.run_episode_with_visualization(delay)
            results.append({'reward': reward, 'steps': steps, 'result': result})
            
            # Pause between episodes
            print(f"Press Enter to continue to next episode (or Ctrl+C to stop)...")
            try:
                input()
            except KeyboardInterrupt:
                print("Stopping visualization...")
                break
        
        # Summary
        print(f"\n=== Summary of {len(results)} episodes ===")
        avg_reward = np.mean([r['reward'] for r in results])
        success_count = sum(1 for r in results if r['result'] == 'SUCCESS')
        print(f"Average reward: {avg_reward:.1f}")
        print(f"Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
        
        return results
    
    def save_episode_animation(self, filename: str, delay: float = 0.05, max_steps: int = 1000):
        """Record and save episode as MP4"""
        from matplotlib.animation import FFMpegWriter
        
        # Setup writer
        writer = FFMpegWriter(fps=int(1/delay), metadata=dict(artist='MoonLander'), bitrate=1800)
        
        with writer.saving(self.visualizer.fig, filename, dpi=100):
            state = self.env.reset()
            self.visualizer.reset_episode()
            self.visualizer.lander_patch.set_color('blue')
            
            episode_reward = 0
            step_count = 0
            
            for step in range(max_steps):
                action, _, _ = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                step_count += 1
                
                state_info = self.env.get_state_info()
                self.visualizer.update_display(state_info, action, step_count, episode_reward)
                
                # Capture frame
                writer.grab_frame()
                
                state = next_state
                if done:
                    self.visualizer.show_episode_result(info.get('landing_result', 'Unknown'), state_info)
                    writer.grab_frame()  # Capture final result frame
                    break


# Usage example
def visualize_trained_agent(checkpoint_path: str, num_episodes: int = 3):
    """Load a trained agent and visualize its performance"""
    from moon_lander_3d import MoonLander3D
    from ppo_trainer import PPOAgent
    
    # Load environment and agent
    env = MoonLander3D()
    agent = PPOAgent(state_dim=env.observation_dim, action_dim=env.action_dim)
    agent.load(checkpoint_path)
    
    # Create visualizer
    visualizer = MoonLanderVisualizer(env)
    runner = VisualTrainingRunner(env, agent, visualizer)
    
    # Run episodes with visualization
    print(f"Visualizing {num_episodes} episodes from: {checkpoint_path}")
    results = runner.run_multiple_episodes(num_episodes, delay=0.02)
    
    runner.save_episode_animation('moon_landing_episode.mp4', delay=0.02)

    plt.show()
    return results

if __name__ == "__main__":
    print("Moon Lander Visualizer ready!")
    print("\nTo use with your trained agent:")
    print("visualize_trained_agent('ppo_checkpoints/ppo_agent_iter_100.pth', num_episodes=3)")
    print("\nTo use during training:")
    print("1. Load your environment and agent")
    print("2. Create visualizer = MoonLanderVisualizer(env)")
    print("3. Create runner = VisualTrainingRunner(env, agent, visualizer)")
    print("4. Run runner.run_episode_with_visualization()")