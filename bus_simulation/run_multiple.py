import numpy as np
from bus_maintenance import BusMaintenanceSimulation, find_max_arrival_rate

def run_multiple_simulations(n_runs=10):
    # Part 1.1 statistics
    inspection_delays = []
    repair_delays = []
    inspection_queue_lengths = []
    repair_queue_lengths = []
    inspection_utilizations = []
    repair_utilizations = []
    
    # Part 1.2 statistics
    max_rates = []
    
    for _ in range(n_runs):
        # Run base simulation
        sim = BusMaintenanceSimulation()
        sim.run_simulation()
        stats = sim.get_statistics()
        
        inspection_delays.append(stats['avg_inspection_delay'])
        repair_delays.append(stats['avg_repair_delay'])
        inspection_queue_lengths.append(stats['avg_inspection_queue_length'])
        repair_queue_lengths.append(stats['avg_repair_queue_length'])
        inspection_utilizations.append(stats['inspection_station_utilization'])
        repair_utilizations.append(stats['repair_station_utilization'])
        
        # Find maximum arrival rate
        results = find_max_arrival_rate()
        if results:
            max_rates.append(1/results[-1][0])
    
    # Calculate statistics
    stats = {
        'inspection_delay': {
            'mean': np.mean(inspection_delays),
            'std': np.std(inspection_delays)
        },
        'repair_delay': {
            'mean': np.mean(repair_delays),
            'std': np.std(repair_delays)
        },
        'inspection_queue_length': {
            'mean': np.mean(inspection_queue_lengths),
            'std': np.std(inspection_queue_lengths)
        },
        'repair_queue_length': {
            'mean': np.mean(repair_queue_lengths),
            'std': np.std(repair_queue_lengths)
        },
        'inspection_utilization': {
            'mean': np.mean(inspection_utilizations),
            'std': np.std(inspection_utilizations)
        },
        'repair_utilization': {
            'mean': np.mean(repair_utilizations),
            'std': np.std(repair_utilizations)
        },
        'max_arrival_rate': {
            'mean': np.mean(max_rates),
            'std': np.std(max_rates)
        }
    }
    
    return stats

if __name__ == "__main__":
    stats = run_multiple_simulations(10)
    
    print("\nSimulation Results (mean ± std):")
    print(f"Inspection queue delay: {stats['inspection_delay']['mean']:.2f} ± {stats['inspection_delay']['std']:.2f} hours")
    print(f"Repair queue delay: {stats['repair_delay']['mean']:.2f} ± {stats['repair_delay']['std']:.2f} hours")
    print(f"Inspection queue length: {stats['inspection_queue_length']['mean']:.2f} ± {stats['inspection_queue_length']['std']:.2f}")
    print(f"Repair queue length: {stats['repair_queue_length']['mean']:.2f} ± {stats['repair_queue_length']['std']:.2f}")
    print(f"Inspection station utilization: {stats['inspection_utilization']['mean']:.2%} ± {stats['inspection_utilization']['std']:.2%}")
    print(f"Repair station utilization: {stats['repair_utilization']['mean']:.2%} ± {stats['repair_utilization']['std']:.2%}")
    print(f"Maximum sustainable arrival rate: {stats['max_arrival_rate']['mean']:.2f} ± {stats['max_arrival_rate']['std']:.2f} buses per hour") 