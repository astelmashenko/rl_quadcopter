import numpy as np
import matplotlib.pyplot as plt


def udacity_distance(x, y):
    return (abs(x - y)).sum()


def eucl_distance(x, y):
    return np.linalg.norm(x - y)


def hover_reward(pose, ang_pose, v, ang_v, target_pose):
    xy_reward = (abs(pose[:2] - target_pose[:2])).sum()
    # reward for to be above target
    z_reward = 2*abs(pose[2] - target_pose[2]).sum()

    z_v = v[2]

    xy_v = v[:2] / 2.
    phi_theta_v = abs(ang_v[:2]).sum()
    ang_pose = abs(ang_pose / (3 * 2 * np.pi)).sum()

    reward = 0

    eucl_dist = eucl_distance(pose, target_pose)
    # penalize for distance increasing
    reward += 1. - 0.3 * eucl_dist

    # z distance
    reward += 1 - 0.3 * z_reward
    # xy distance
    reward += 1 - 0.3 * xy_reward

    # velocity
    # z velocity
    reward += z_v
    # xy velocity
    #reward -= abs(xy_v).sum()

    # angles
    reward -= 5 * ang_pose

    # angle velocities
    #reward -= phi_theta_v

    return np.clip(reward, -2, 2)


def plots(results):
    plt.figure(figsize=(18, 18))

    plt.subplot(3, 3, 1)
    plt.grid(True)
    plt.title('distance')
    plt.plot(results['time'], results['distance'], label='distance')
    plt.legend()
    _ = plt.ylim()

    plt.subplot(3, 3, 2)
    plt.title('x, y, z')
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.grid(True)
    plt.legend()
    _ = plt.ylim()

    plt.subplot(3, 3, 3)
    plt.grid(True)
    plt.title('episode reward')
    plt.plot(results['time'], results['reward'], label='episode reward')
    plt.legend()
    _ = plt.ylim()

    plt.subplot(3, 3, 4)
    plt.grid(True)
    plt.title('xyz velocities')
    plt.plot(results['time'], results['x_velocity'], label='x_hat')
    plt.plot(results['time'], results['y_velocity'], label='y_hat')
    plt.plot(results['time'], results['z_velocity'], label='z_hat')
    plt.legend()
    _ = plt.ylim()

    plt.subplot(3, 3, 5)
    plt.grid(True)
    plt.title('angles')
    plt.plot(results['time'], results['phi'], label='phi')
    plt.plot(results['time'], results['theta'], label='theta')
    plt.plot(results['time'], results['psi'], label='psi')
    plt.legend()
    _ = plt.ylim()

    plt.subplot(3, 3, 6)
    plt.grid(True)
    plt.title('angle velocities')
    plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    plt.legend()
    _ = plt.ylim()

    plt.subplot(3, 3, 7)
    plt.grid(True)
    plt.title('Rotor speeds')
    plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    plt.legend()
    _ = plt.ylim()
