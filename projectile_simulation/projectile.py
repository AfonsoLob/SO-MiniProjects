import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import argparse

@dataclass
class ProjectileState:
    x: float  # x position
    z: float  # z position
    vx: float  # x velocity
    vz: float  # z velocity

class ProjectileSimulation:
    def __init__(self, x0: float, z0: float, vx0: float, vz0: float, 
                 u: float, m: float, g: float, dt: float, t_final: float):
        self.state = ProjectileState(x0, z0, vx0, vz0)
        self.u = u
        self.m = m
        self.g = g
        self.dt = dt
        self.t_final = t_final
        self.time_points = np.arange(0, t_final + dt, dt)
        self.history: List[ProjectileState] = [ProjectileState(x0, z0, vx0, vz0)]

    def acceleration(self, state: ProjectileState) -> Tuple[float, float]:
        """Calculate acceleration components based on current state"""
        v = np.sqrt(state.vx**2 + state.vz**2)
        ax = -self.u * state.vx * abs(state.vx) / self.m
        az = -self.g - self.u * state.vz * abs(state.vz) / self.m
        return ax, az

    def forward_euler_step(self, state: ProjectileState) -> ProjectileState:
        """Perform one step of Forward Euler integration"""
        ax, az = self.acceleration(state)
        new_vx = state.vx + ax * self.dt
        new_vz = state.vz + az * self.dt
        new_x = state.x + state.vx * self.dt
        new_z = state.z + state.vz * self.dt
        return ProjectileState(new_x, new_z, new_vx, new_vz)

    def runge_kutta_step(self, state: ProjectileState) -> ProjectileState:
        """Perform one step of Runge-Kutta 4th order integration"""
        # k1
        ax1, az1 = self.acceleration(state)
        k1_vx = ax1 * self.dt
        k1_vz = az1 * self.dt
        k1_x = state.vx * self.dt
        k1_z = state.vz * self.dt

        # k2
        state2 = ProjectileState(
            state.x + k1_x/2,
            state.z + k1_z/2,
            state.vx + k1_vx/2,
            state.vz + k1_vz/2
        )
        ax2, az2 = self.acceleration(state2)
        k2_vx = ax2 * self.dt
        k2_vz = az2 * self.dt
        k2_x = state2.vx * self.dt
        k2_z = state2.vz * self.dt

        # k3
        state3 = ProjectileState(
            state.x + k2_x/2,
            state.z + k2_z/2,
            state.vx + k2_vx/2,
            state.vz + k2_vz/2
        )
        ax3, az3 = self.acceleration(state3)
        k3_vx = ax3 * self.dt
        k3_vz = az3 * self.dt
        k3_x = state3.vx * self.dt
        k3_z = state3.vz * self.dt

        # k4
        state4 = ProjectileState(
            state.x + k3_x,
            state.z + k3_z,
            state.vx + k3_vx,
            state.vz + k3_vz
        )
        ax4, az4 = self.acceleration(state4)
        k4_vx = ax4 * self.dt
        k4_vz = az4 * self.dt
        k4_x = state4.vx * self.dt
        k4_z = state4.vz * self.dt

        # Final update
        new_x = state.x + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        new_z = state.z + (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6
        new_vx = state.vx + (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6
        new_vz = state.vz + (k1_vz + 2*k2_vz + 2*k3_vz + k4_vz) / 6

        return ProjectileState(new_x, new_z, new_vx, new_vz)

    def simulate(self, method: str = 'euler'):
        """Run the simulation using specified method"""
        self.history = [self.state]
        
        for _ in range(len(self.time_points) - 1):
            if method == 'euler':
                self.state = self.forward_euler_step(self.state)
            else:  # runge-kutta
                self.state = self.runge_kutta_step(self.state)
            self.history.append(ProjectileState(
                self.state.x, self.state.z, self.state.vx, self.state.vz
            ))

    def get_results(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays of x, z positions and velocities"""
        x = np.array([state.x for state in self.history])
        z = np.array([state.z for state in self.history])
        vx = np.array([state.vx for state in self.history])
        vz = np.array([state.vz for state in self.history])
        return x, z, vx, vz

def compare_methods(x0: float, z0: float, vx0: float, vz0: float,
                   u: float, m: float, g: float, dt: float, t_final: float):
    """Compare Forward Euler and Runge-Kutta methods"""
    # Run both simulations
    sim_euler = ProjectileSimulation(x0, z0, vx0, vz0, u, m, g, dt, t_final)
    sim_rk = ProjectileSimulation(x0, z0, vx0, vz0, u, m, g, dt, t_final)
    
    sim_euler.simulate(method='euler')
    sim_rk.simulate(method='rk')
    
    # Get results
    x_euler, z_euler, vx_euler, vz_euler = sim_euler.get_results()
    x_rk, z_rk, vx_rk, vz_rk = sim_rk.get_results()
    
    # Calculate differences
    x_diff = np.abs(x_euler - x_rk)
    z_diff = np.abs(z_euler - z_rk)
    vx_diff = np.abs(vx_euler - vx_rk)
    vz_diff = np.abs(vz_euler - vz_rk)
    
    return {
        'max_x_diff': np.max(x_diff),
        'max_z_diff': np.max(z_diff),
        'max_vx_diff': np.max(vx_diff),
        'max_vz_diff': np.max(vz_diff),
        'euler': (x_euler, z_euler, vx_euler, vz_euler),
        'rk': (x_rk, z_rk, vx_rk, vz_rk)
    }

def main():
    parser = argparse.ArgumentParser(description='Projectile Motion Simulation')
    parser.add_argument('--x0', type=float, default=0.0, help='Initial x position')
    parser.add_argument('--z0', type=float, default=0.0, help='Initial z position')
    parser.add_argument('--vx0', type=float, default=10.0, help='Initial x velocity')
    parser.add_argument('--vz0', type=float, default=10.0, help='Initial z velocity')
    parser.add_argument('--u', type=float, default=0.1, help='Air resistance coefficient')
    parser.add_argument('--m', type=float, default=1.0, help='Mass')
    parser.add_argument('--g', type=float, default=9.81, help='Gravity')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    parser.add_argument('--t_final', type=float, default=10.0, help='Final time')
    parser.add_argument('--method', choices=['euler', 'rk', 'compare'], default='compare',
                      help='Integration method to use')
    
    args = parser.parse_args()
    
    if args.method == 'compare':
        results = compare_methods(
            args.x0, args.z0, args.vx0, args.vz0,
            args.u, args.m, args.g, args.dt, args.t_final
        )
        print("\nComparison Results:")
        print(f"Maximum difference in x position: {results['max_x_diff']:.6f}")
        print(f"Maximum difference in z position: {results['max_z_diff']:.6f}")
        print(f"Maximum difference in x velocity: {results['max_vx_diff']:.6f}")
        print(f"Maximum difference in z velocity: {results['max_vz_diff']:.6f}")
    else:
        sim = ProjectileSimulation(
            args.x0, args.z0, args.vx0, args.vz0,
            args.u, args.m, args.g, args.dt, args.t_final
        )
        sim.simulate(method=args.method)
        x, z, vx, vz = sim.get_results()
        
        print("\nSimulation Results:")
        print(f"Final x position: {x[-1]:.6f}")
        print(f"Final z position: {z[-1]:.6f}")
        print(f"Final x velocity: {vx[-1]:.6f}")
        print(f"Final z velocity: {vz[-1]:.6f}")

if __name__ == "__main__":
    main() 