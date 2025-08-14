import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = 6.6743e-11 #Gravitational Constant (m^3 kg^-1 s^-2)

def compute_accel(positions, masses):
    N = len(masses) # Number of bodies
    acc = np.zeros_like(positions) # Initial acceleration as zero
    for i in range(N):
        for j in range(N):
            if i != j: # Calculating force given sum of all bodies according to newton's law of gravitation
                r_vec = positions[j] - positions[i] # Vector from body i to body j
                dist = np.linalg.norm(r_vec) # Distance between bodies
                acc[i] += G * masses[j] * r_vec / dist**3 # Acceleration of body i due to body j
    return acc

def velocity_verlet(positions, velocities, masses, dt, steps): # Computes and updates positions and velocities over time
    positions_list = [positions.copy()]
    velocities_list = [velocities.copy()]

    acc = compute_accel(positions, masses)

    for _ in range(steps):
        positions += velocities * dt + 0.5 * acc * dt**2
        # print(positions)
        new_acc = compute_accel(positions,masses) # Compute new acceleration after updating positions
        velocities += 0.5* (acc + new_acc) * dt 
        acc = new_acc

        positions_list.append(positions.copy())
        velocities_list.append(velocities.copy())

    return np.array(positions_list), np.array(velocities_list)

masses = np.array([
    1.9e30,        # Sun
    5.972e24,      # Earth
    6.39e23,       # Mars
    4.867e24,      # Venus
    1.898e27,      # Jupiter
    7.34767309e22  # Moon
], dtype=np.float64)

positions = np.array([
    [0, 0],                       # Sun
    [1.496e11, 0],                # Earth
    [2.279e11, 0],                # Mars
    [1.082e11, 0],                # Venus
    [7.785e11, 0],                # Jupiter
    [1.496e11 + 3.844e8, 0]       # Moon (relative to Earth)
], dtype=np.float64)

velocities = np.array([
    [0, 5000],                       # Sun
    [0, 29780],                   # Earth
    [0, 24077],                   # Mars
    [0, 35020],                   # Venus
    [0, 13070],                   # Jupiter
    [0, 29780 + 1022]             # Moon (Earth's velocity + Moon's orbital velocity)
], dtype=np.float64)


pos_hist, vel_hist = velocity_verlet(positions, velocities, masses, dt=60*60*6, steps=50000)

labels = ["Sun", "Earth", "Mars", "Venus", "Jupiter", "Moon"]
colors = ["yellow", "blue", "red", "gray", "green", "lightgray"]

fig, ax = plt.subplots(figsize=(8,8))
lines = [ax.plot([], [], 'o', color=colors[i], label=labels[i])[0] for i in range(len(labels))]
ax.set_xlim(-8e11, 8e11)
ax.set_ylim(-8e11, 8e11)
ax.set_xlabel('x position (m)')
ax.set_ylabel('y position (m)')
ax.legend()
ax.set_title('OrbSim')
ax.set_aspect('equal')

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    for i, line in enumerate(lines):
        line.set_data(pos_hist[:frame+1, i, 0], pos_hist[:frame+1, i, 1])
    return lines

ani = animation.FuncAnimation(fig, update, frames=len(pos_hist), init_func=init, blit=True, interval=20)
plt.show()

