import numpy as np
from projectile import ProjectileSimulation, compare_methods
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def run_comparison_tests(n_runs=10):
    # Test parameters
    test_cases = [
        {
            'name': 'High Velocity',
            'params': {
                'x0': 0, 'z0': 0,
                'vx0': 50, 'vz0': 50,
                'u': 0.1, 'm': 1.0,
                'g': 9.81, 'dt': 0.01,
                't_final': 10.0
            }
        },
        {
            'name': 'Low Velocity',
            'params': {
                'x0': 0, 'z0': 0,
                'vx0': 10, 'vz0': 10,
                'u': 0.1, 'm': 1.0,
                'g': 9.81, 'dt': 0.01,
                't_final': 5.0
            }
        },
        {
            'name': 'High Air Resistance',
            'params': {
                'x0': 0, 'z0': 0,
                'vx0': 30, 'vz0': 30,
                'u': 0.5, 'm': 1.0,
                'g': 9.81, 'dt': 0.01,
                't_final': 8.0
            }
        }
    ]
    
    results = {}
    
    for case in test_cases:
        print(f"\nRunning {case['name']} test case...")
        case_results = []
        
        for _ in range(n_runs):
            result = compare_methods(**case['params'])
            case_results.append(result)
        
        # Calculate statistics
        x_diffs = [r['max_x_diff'] for r in case_results]
        z_diffs = [r['max_z_diff'] for r in case_results]
        vx_diffs = [r['max_vx_diff'] for r in case_results]
        vz_diffs = [r['max_vz_diff'] for r in case_results]
        
        results[case['name']] = {
            'x_diff': {'mean': np.mean(x_diffs), 'std': np.std(x_diffs)},
            'z_diff': {'mean': np.mean(z_diffs), 'std': np.std(z_diffs)},
            'vx_diff': {'mean': np.mean(vx_diffs), 'std': np.std(vx_diffs)},
            'vz_diff': {'mean': np.mean(vz_diffs), 'std': np.std(vz_diffs)}
        }
        
        # Plot trajectories for the last run
        plt.figure(figsize=(10, 6))
        x_euler, z_euler, _, _ = case_results[-1]['euler']
        x_rk, z_rk, _, _ = case_results[-1]['rk']
        
        plt.plot(x_euler, z_euler, 'b-', label='Euler')
        plt.plot(x_rk, z_rk, 'r--', label='Runge-Kutta')
        plt.title(f'Trajectory Comparison - {case["name"]}')
        plt.xlabel('x position')
        plt.ylabel('z position')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{case["name"].lower().replace(" ", "_")}_trajectory.png')
        plt.close()

        # Generate differences plot (position and velocity)
        t = np.arange(0, len(x_euler)) * case['params']['dt']
        vx_euler, vz_euler, vx_rk, vz_rk = case_results[-1]['euler'][2], case_results[-1]['euler'][3], case_results[-1]['rk'][2], case_results[-1]['rk'][3]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # Position differences
        ax1.plot(t, np.array(x_euler) - np.array(x_rk), label='x difference')
        ax1.plot(t, np.array(z_euler) - np.array(z_rk), label='z difference')
        ax1.set_title('Position Differences (Euler - RK4)')
        ax1.set_ylabel('Position Difference (m)')
        ax1.legend()
        ax1.grid(True)
        # Velocity differences
        ax2.plot(t, np.array(vx_euler) - np.array(vx_rk), label='vx difference')
        ax2.plot(t, np.array(vz_euler) - np.array(vz_rk), label='vz difference')
        ax2.set_title('Velocity Differences (Euler - RK4)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity Difference (m/s)')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(f'differences_{case["name"].lower().replace(" ", "_")}.png')
        plt.close()
    
    return results

if __name__ == "__main__":
    results = run_comparison_tests(n_runs=10)
    
    print("\nComparison Results (mean ± std):")
    for case_name, case_results in results.items():
        print(f"\n{case_name}:")
        print(f"Maximum x position difference: {case_results['x_diff']['mean']:.6f} ± {case_results['x_diff']['std']:.6f}")
        print(f"Maximum z position difference: {case_results['z_diff']['mean']:.6f} ± {case_results['z_diff']['std']:.6f}")
        print(f"Maximum x velocity difference: {case_results['vx_diff']['mean']:.6f} ± {case_results['vx_diff']['std']:.6f}")
        print(f"Maximum z velocity difference: {case_results['vz_diff']['mean']:.6f} ± {case_results['vz_diff']['std']:.6f}") 