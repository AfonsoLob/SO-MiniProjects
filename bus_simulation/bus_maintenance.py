import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import random

@dataclass
class Bus:
    arrival_time: float
    inspection_start_time: float = 0
    inspection_end_time: float = 0
    repair_start_time: float = 0
    repair_end_time: float = 0
    needs_repair: bool = False

class BusMaintenanceSimulation:
    def __init__(self, mean_interarrival_time: float = 2.0):
        self.mean_interarrival_time = mean_interarrival_time
        self.inspection_queue: List[Bus] = []
        self.repair_queue: List[Bus] = []
        self.buses: List[Bus] = []
        self.inspection_station_busy = False
        self.repair_stations_busy = [False, False]
        self.current_time = 0
        self.simulation_time = 160  # hours
        
        # Statistics
        self.inspection_queue_delays = []
        self.repair_queue_delays = []
        self.inspection_queue_lengths = []
        self.repair_queue_lengths = []
        self.inspection_busy_time = 0
        self.repair_busy_time = 0

    def generate_interarrival_time(self) -> float:
        return random.expovariate(1.0 / self.mean_interarrival_time)

    def generate_inspection_time(self) -> float:
        return random.uniform(0.25, 1.05)  # 15 minutes to 1.05 hours

    def generate_repair_time(self) -> float:
        return random.uniform(2.1, 4.5)

    def needs_repair(self) -> bool:
        return random.random() < 0.3  # 30% probability

    def run_simulation(self):
        next_arrival_time = self.generate_interarrival_time()
        
        while self.current_time < self.simulation_time:
            # Update statistics
            self.inspection_queue_lengths.append(len(self.inspection_queue))
            self.repair_queue_lengths.append(len(self.repair_queue))
            
            # Check for next event
            events = []
            if next_arrival_time < self.simulation_time:
                events.append(('arrival', next_arrival_time))
            
            # Check inspection completion
            for bus in self.buses:
                if bus.inspection_end_time > self.current_time and bus.inspection_end_time < self.simulation_time:
                    events.append(('inspection_complete', bus.inspection_end_time))
                if bus.repair_end_time > self.current_time and bus.repair_end_time < self.simulation_time:
                    events.append(('repair_complete', bus.repair_end_time))
            
            if not events:
                break
                
            # Process next event
            event_type, event_time = min(events, key=lambda x: x[1])
            self.current_time = event_time
            
            if event_type == 'arrival':
                # New bus arrives
                bus = Bus(arrival_time=self.current_time)
                self.buses.append(bus)
                self.inspection_queue.append(bus)
                next_arrival_time = self.current_time + self.generate_interarrival_time()
                
            elif event_type == 'inspection_complete':
                # Inspection completed
                self.inspection_station_busy = False
                for bus in self.buses:
                    if bus.inspection_end_time == self.current_time:
                        if self.needs_repair():
                            bus.needs_repair = True
                            self.repair_queue.append(bus)
                        break
                        
            elif event_type == 'repair_complete':
                # Repair completed
                for i, busy in enumerate(self.repair_stations_busy):
                    if busy:
                        self.repair_stations_busy[i] = False
                        break
                        
            # Start new inspections if possible
            if not self.inspection_station_busy and self.inspection_queue:
                bus = self.inspection_queue.pop(0)
                bus.inspection_start_time = self.current_time
                inspection_time = self.generate_inspection_time()
                bus.inspection_end_time = self.current_time + inspection_time
                self.inspection_station_busy = True
                self.inspection_queue_delays.append(bus.inspection_start_time - bus.arrival_time)
                
            # Start new repairs if possible
            if self.repair_queue:
                for i, busy in enumerate(self.repair_stations_busy):
                    if not busy and self.repair_queue:
                        bus = self.repair_queue.pop(0)
                        bus.repair_start_time = self.current_time
                        repair_time = self.generate_repair_time()
                        bus.repair_end_time = self.current_time + repair_time
                        self.repair_stations_busy[i] = True
                        self.repair_queue_delays.append(bus.repair_start_time - bus.inspection_end_time)
                        
            # Update busy times
            if self.inspection_station_busy:
                self.inspection_busy_time += self.current_time - self.current_time
            self.repair_busy_time += sum(1 for busy in self.repair_stations_busy if busy) * (self.current_time - self.current_time)

    def get_statistics(self) -> dict:
        return {
            'avg_inspection_delay': np.mean(self.inspection_queue_delays) if self.inspection_queue_delays else 0,
            'avg_repair_delay': np.mean(self.repair_queue_delays) if self.repair_queue_delays else 0,
            'avg_inspection_queue_length': np.mean(self.inspection_queue_lengths),
            'avg_repair_queue_length': np.mean(self.repair_queue_lengths),
            'inspection_station_utilization': self.inspection_busy_time / self.simulation_time,
            'repair_station_utilization': (self.repair_busy_time / 2) / self.simulation_time
        }

def find_max_arrival_rate():
    mean_interarrival_times = np.linspace(0.5, 2.0, 31)  # Test from 0.5 to 2.0 hours
    results = []
    
    for mean_time in mean_interarrival_times:
        sim = BusMaintenanceSimulation(mean_interarrival_time=mean_time)
        sim.run_simulation()
        stats = sim.get_statistics()
        
        # Check if system is stable (queues don't grow unbounded)
        if stats['avg_inspection_queue_length'] < 100 and stats['avg_repair_queue_length'] < 100:
            results.append((mean_time, stats))
        else:
            break
            
    return results

if __name__ == "__main__":
    # Part 1.1
    sim = BusMaintenanceSimulation()
    sim.run_simulation()
    stats = sim.get_statistics()
    print("\nPart 1.1 Results:")
    print(f"Average inspection queue delay: {stats['avg_inspection_delay']:.2f} hours")
    print(f"Average repair queue delay: {stats['avg_repair_delay']:.2f} hours")
    print(f"Average inspection queue length: {stats['avg_inspection_queue_length']:.2f}")
    print(f"Average repair queue length: {stats['avg_repair_queue_length']:.2f}")
    print(f"Inspection station utilization: {stats['inspection_station_utilization']:.2%}")
    print(f"Repair station utilization: {stats['repair_station_utilization']:.2%}")
    
    # Part 1.2
    print("\nPart 1.2 Results:")
    results = find_max_arrival_rate()
    if results:
        max_rate = results[-1][0]
        print(f"Maximum sustainable arrival rate: {1/max_rate:.2f} buses per hour")
        print(f"(Minimum mean interarrival time: {max_rate:.2f} hours)") 