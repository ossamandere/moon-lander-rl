import numpy as np
import math
from typing import Tuple, Dict, Any, Optional

class MoonLander3D:
    """
    3D Moon Lander Physics Environment
    Following OpenAI Gym interface for RL training
    """
    
    def __init__(self, 
                 target_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 initial_position: Tuple[float, float, float] = (0.0, 0.0, 100.0),
                 initial_velocity: Tuple[float, float, float] = (5.0, 5.0, -10.0),
                 dt: float = 0.02,
                 max_episode_steps: int = 1000):
        """
        Initialize the moon lander environment
        
        Args:
            target_position: Target landing coordinates (x, y, z)
            initial_position: Starting position (x, y, z)
            initial_velocity: Starting velocity (vx, vy, vz)
            dt: Physics timestep in seconds
            max_episode_steps: Maximum steps before episode termination
        """
        # Physical constants
        self.LUNAR_GRAVITY = -1.62  # m/s^2 (negative = downward)
        self.MAX_THRUST_TO_WEIGHT = 1.8  # Maximum thrust relative to Earth weight
        self.MAX_GIMBAL_ANGLE = math.radians(15)  # 15 degrees in radians
        
        # Lander specifications
        self.DRY_MASS = 15000.0  # kg (dry mass without fuel)
        self.INITIAL_FUEL_MASS = 8000.0  # kg
        self.SPECIFIC_IMPULSE = 311.0  # seconds (efficiency of engine)
        self.G0 = 9.80665  # m/s^2 (standard gravity for Isp calculations)
        
        # Success criteria
        self.MAX_LANDING_VELOCITY_LATERAL = 0.5  # m/s
        self.MAX_LANDING_VELOCITY_VERTICAL = 2.0   # m/s
        
        # Environment parameters
        self.target_position = np.array(target_position, dtype=np.float32)
        self.initial_position = np.array(initial_position, dtype=np.float32)
        self.initial_velocity = np.array(initial_velocity, dtype=np.float32)
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        
        # State variables (will be set in reset())
        self.position = None
        self.velocity = None
        self.fuel_mass = None
        self.step_count = None
        self.done = None
        
        # Action and observation space dimensions
        self.action_dim = 3  # [thrust_magnitude, gimbal_x, gimbal_y]
        self.observation_dim = 11  # [pos(3), vel(3), target_dist(1), mass(1), fuel(1), fuel_ratio(1), steps_remaining(1)]
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.position = self.initial_position.copy()
        self.velocity = self.initial_velocity.copy()
        self.fuel_mass = self.INITIAL_FUEL_MASS
        self.step_count = 0
        self.done = False
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step
        
        Args:
            action: [thrust_magnitude, gimbal_x, gimbal_y]
                   - thrust_magnitude: 0.0 to 1.0 (fraction of max thrust)
                   - gimbal_x, gimbal_y: -1.0 to 1.0 (gimbal angles normalized)
        
        Returns:
            observation, reward, done, info
        """
        if self.done:
            return self._get_observation(), 0.0, True, {}
        
        # Parse and clamp actions
        thrust_fraction = np.clip(action[0], 0.0, 1.0)
        gimbal_x = np.clip(action[1], -1.0, 1.0) * self.MAX_GIMBAL_ANGLE
        gimbal_y = np.clip(action[2], -1.0, 1.0) * self.MAX_GIMBAL_ANGLE
        
        # Calculate current mass and maximum thrust
        current_mass = self.DRY_MASS + self.fuel_mass
        max_thrust = current_mass * abs(self.LUNAR_GRAVITY) * self.MAX_THRUST_TO_WEIGHT
        
        # Calculate actual thrust magnitude
        actual_thrust = thrust_fraction * max_thrust if self.fuel_mass > 0 else 0.0
        
        # Calculate thrust vector (accounting for gimbal)
        # Default thrust direction is upward (0, 0, 1)
        thrust_direction = np.array([
            math.sin(gimbal_x) * math.cos(gimbal_y),
            math.sin(gimbal_y),
            math.cos(gimbal_x) * math.cos(gimbal_y)
        ])
        thrust_vector = actual_thrust * thrust_direction
        
        # Calculate fuel consumption
        if actual_thrust > 0 and self.fuel_mass > 0:
            # Fuel flow rate based on thrust and specific impulse
            fuel_flow_rate = actual_thrust / (self.SPECIFIC_IMPULSE * self.G0)
            fuel_consumed = fuel_flow_rate * self.dt
            self.fuel_mass = max(0.0, self.fuel_mass - fuel_consumed)
        
        # Update physics
        current_mass = self.DRY_MASS + self.fuel_mass
        
        # Acceleration from thrust
        thrust_acceleration = thrust_vector / current_mass if current_mass > 0 else np.zeros(3)
        
        # Acceleration from gravity (only affects z-component)
        gravity_acceleration = np.array([0.0, 0.0, self.LUNAR_GRAVITY])
        
        # Total acceleration
        total_acceleration = thrust_acceleration + gravity_acceleration
        
        # Update velocity and position (Euler integration)
        self.velocity += total_acceleration * self.dt
        self.position += self.velocity * self.dt
        
        # Update step counter
        self.step_count += 1
        
        # Check termination conditions
        reward, self.done, info = self._calculate_reward_and_termination()
        
        return self._get_observation(), reward, self.done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        target_distance = np.linalg.norm(self.position - self.target_position)
        current_mass = self.DRY_MASS + self.fuel_mass
        fuel_ratio = self.fuel_mass / self.INITIAL_FUEL_MASS
        steps_remaining = (self.max_episode_steps - self.step_count) / self.max_episode_steps
        
        observation = np.concatenate([
            self.position,                    # 3 components
            self.velocity,                    # 3 components
            [target_distance],                # 1 component
            [current_mass],                   # 1 component
            [self.fuel_mass],                 # 1 component
            [fuel_ratio],                     # 1 component
            [steps_remaining]                 # 1 component
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_reward_and_termination(self) -> Tuple[float, bool, Dict[str, Any]]:
        """Calculate reward and check termination conditions"""
        info = {}
        
        # Check if crashed (hit ground with too much velocity or wrong position)
        if self.position[2] <= 0:  # Hit ground (z <= 0)
            lateral_velocity = np.linalg.norm(self.velocity[:2])  # x, y velocity magnitude
            vertical_velocity = abs(self.velocity[2])  # z velocity magnitude
            
            # Check if landing is successful
            if (lateral_velocity <= self.MAX_LANDING_VELOCITY_LATERAL and 
                vertical_velocity <= self.MAX_LANDING_VELOCITY_VERTICAL):
                # Successful landing!
                fuel_bonus = (self.fuel_mass / self.INITIAL_FUEL_MASS) * 100  # Bonus for remaining fuel
                reward = 1000 + fuel_bonus
                info['landing_result'] = 'SUCCESS'
                return reward, True, info
            else:
                # Crashed - too fast landing
                reward = -1000
                info['landing_result'] = 'CRASH'
                info['lateral_velocity'] = lateral_velocity
                info['vertical_velocity'] = vertical_velocity
                return reward, True, info
        
        # Check if too far from target (safety boundary)
        distance_to_target = np.linalg.norm(self.position - self.target_position)
        if distance_to_target > 500:  # 500m safety boundary
            reward = -500
            info['landing_result'] = 'OUT_OF_BOUNDS'
            return reward, True, info
        
        # Check if max steps reached
        if self.step_count >= self.max_episode_steps:
            reward = -200
            info['landing_result'] = 'TIMEOUT'
            return reward, True, info
        
        # Ongoing reward shaping
        reward = self._calculate_shaped_reward(distance_to_target)
        return reward, False, info
    
    def _calculate_shaped_reward(self, distance_to_target: float) -> float:
        """Calculate shaped reward for ongoing flight"""
        # Distance reward (closer to target is better)
        distance_reward = -distance_to_target * 0.1
        
        # Velocity penalty (discourage excessive speed)
        velocity_magnitude = np.linalg.norm(self.velocity)
        velocity_penalty = -velocity_magnitude * 0.01
        
        # Small step penalty to encourage efficiency
        step_penalty = -0.1
        
        total_reward = distance_reward + velocity_penalty + step_penalty
        return total_reward
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed state information for debugging/analysis"""
        current_mass = self.DRY_MASS + self.fuel_mass
        distance_to_target = np.linalg.norm(self.position - self.target_position)
        
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'mass': current_mass,
            'fuel_mass': self.fuel_mass,
            'fuel_ratio': self.fuel_mass / self.INITIAL_FUEL_MASS,
            'distance_to_target': distance_to_target,
            'step_count': self.step_count,
            'altitude': self.position[2]
        }
    
    def render(self, mode='human') -> Optional[np.ndarray]:
        """Basic text rendering - will be replaced with 3D renderer later"""
        if mode == 'human':
            state = self.get_state_info()
            print(f"Step: {self.step_count:4d} | "
                  f"Pos: ({state['position'][0]:6.1f}, {state['position'][1]:6.1f}, {state['position'][2]:6.1f}) | "
                  f"Vel: ({state['velocity'][0]:5.1f}, {state['velocity'][1]:5.1f}, {state['velocity'][2]:5.1f}) | "
                  f"Fuel: {state['fuel_ratio']:4.1%} | "
                  f"Dist: {state['distance_to_target']:6.1f}m")
        
        return None


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = MoonLander3D()
    
    print("Moon Lander 3D Environment Test")
    print("=" * 50)
    print(f"Action dimensions: {env.action_dim}")
    print(f"Observation dimensions: {env.observation_dim}")
    print()
    
    # Test with random actions
    obs = env.reset()
    total_reward = 0
    
    for step in range(100):
        # Random action: [thrust_fraction, gimbal_x, gimbal_y]
        action = np.array([
            np.random.random() * 0.5,  # Moderate thrust
            (np.random.random() - 0.5) * 0.2,  # Small gimbal adjustments
            (np.random.random() - 0.5) * 0.2
        ])
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:  # Render every 10 steps
            env.render()
        
        if done:
            print(f"\nEpisode finished at step {step}")
            print(f"Result: {info.get('landing_result', 'Unknown')}")
            print(f"Total reward: {total_reward:.2f}")
            break
    
    print("\nFinal state:")
    final_state = env.get_state_info()
    for key, value in final_state.items():
        print(f"  {key}: {value}")