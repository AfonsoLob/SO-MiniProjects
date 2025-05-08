# Simulation and Optimization Project

This project contains two simulation programs:
1. Bus Maintenance Facility Simulation
2. Projectile Motion Simulation

## Requirements

- Python 3.8+
- NumPy

Install the required packages:
```bash
pip install numpy
```

## Bus Maintenance Facility Simulation

This program simulates a bus maintenance facility with one inspection station and two repair stations. It calculates various statistics about queue delays, queue lengths, and station utilization.

### Usage

```bash
python bus_simulation/bus_maintenance.py
```

The program will:
1. Run a 160-hour simulation with the default parameters
2. Calculate and display:
   - Average inspection queue delay
   - Average repair queue delay
   - Average inspection queue length
   - Average repair queue length
   - Inspection station utilization
   - Repair station utilization
3. Find the maximum sustainable bus arrival rate

## Projectile Motion Simulation

This program simulates projectile motion with air resistance using either Forward Euler or Runge-Kutta integration methods.

### Usage

```bash
python projectile_simulation/projectile.py --method compare
```

Options:
- `--x0`: Initial x position (default: 0.0)
- `--z0`: Initial z position (default: 0.0)
- `--vx0`: Initial x velocity (default: 10.0)
- `--vz0`: Initial z velocity (default: 10.0)
- `--u`: Air resistance coefficient (default: 0.1)
- `--m`: Mass (default: 1.0)
- `--g`: Gravity (default: 9.81)
- `--dt`: Time step (default: 0.01)
- `--t_final`: Final time (default: 10.0)
- `--method`: Integration method ('euler', 'rk', or 'compare') (default: 'compare')

Example:
```bash
python projectile_simulation/projectile.py --x0 0 --z0 100 --vx0 20 --vz0 0 --u 0.1 --m 1 --g 9.81 --dt 0.01 --t_final 10 --method compare
```

When using the 'compare' method, the program will show the maximum differences between the Forward Euler and Runge-Kutta methods for position and velocity components.

## Project Structure

```
.
├── README.md
├── bus_simulation/
│   └── bus_maintenance.py
└── projectile_simulation/
    └── projectile.py
```