import pandas as pd
import matplotlib.pyplot as plt


def plot_cycle_curve(df, cycle_idx):
    """查看任意一圈的电流电压曲线"""
    cycle_df = df[df['cycle_number'] == cycle_idx]
    if cycle_df.empty:
        print(f"Warning: Cycle {cycle_idx} not found in the data.")
        return

    _, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)', color=color)
    ax1.plot(cycle_df['time'], cycle_df['voltage'], color=color, label='Voltage')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Current (A)', color=color)  
    ax2.plot(cycle_df['time'], cycle_df['current'], color=color, linestyle='--', label='Current')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Voltage and Current Plots - Cycle {cycle_idx}')
    plt.tight_layout()
    plt.show()


def plot_capacity_degradation(df):
    """查看整个电池的容量衰减曲线"""
    capacity = df.groupby('cycle_number')['capacity'].max()
    
    plt.figure(figsize=(10, 5))
    plt.plot(capacity.index, capacity.values, marker='o', markersize=4, linestyle='-', color='tab:green')
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (Ah)')
    plt.title('Battery Capacity Degradation Curve')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()