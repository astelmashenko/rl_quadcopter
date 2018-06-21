import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pow


def udacity_distance(x, y):
    return (abs(x - y)).sum()


def eucl_distance(x, y):
    return np.linalg.norm(x - y)


def hover_reward(pose, ang_pose, v, ang_v, target_pose):
    xy_reward = abs(pose[:2] - target_pose[:2]).sum()
    # reward for to be above target
    x_reward = pow(abs(pose[0] - target_pose[0]).sum(), 1/4)
    y_reward = pow(abs(pose[1] - target_pose[1]).sum(), 1/4)
    z_reward = pow(abs(pose[2] - target_pose[2]).sum(), 1/4)

    z_v = pow(abs(v[2]), 1/4)

    xy_v = v[:2] / 2.
    ang_xy_v = pow(abs(ang_v[:2]).sum(), 1/4)
    ang_pose = pow(abs(ang_pose / (3 * 2 * np.pi)).sum(), 1/4)

    reward = 0
    # np.clip(, -1, 1)
    # eucl_dist = eucl_distance(pose, target_pose)
    # penalize for distance increasing
    # reward += np.clip(5 * (1. - 2. * eucl_dist), -5, 5)

    # z distance
    reward += np.clip((5 - 1.9 * z_reward), -0.4, 1.5)
    # xy distance
    # reward += np.clip(5 * (1 - 2. * xy_reward), -4, 3)
    reward += np.clip((5 - 1.9 * x_reward), -0.4, 1.5)
    reward += np.clip((5 - 1.9 * y_reward), -0.4, 1.5)

    # velocity
    # z velocity
    reward += np.clip((5 - 2 * z_v), -0.2, 2)  # np.clip(10 * (1 - 3.0 * z_v), 10, -10)
    # xy velocity
    # reward -= abs(xy_v).sum()

    # angles
    reward -= np.clip(5 - 2 * ang_pose, -0.2, 1)

    # angle velocities
    # reward -= np.clip(ang_xy_v, -1, 1)
    # reward += np.clip(1 - 2 * ang_xy_v, -1, 1)

    # return np.clip(reward, -0.2, 10)
    return reward


if __name__ == '__main__':
    data = pd.read_csv('data1.csv')
    cols = ['x', 'y', 'z',
            'phi', 'theta', 'psi',
            'x_velocity', 'y_velocity', 'z_velocity',
            'phi_velocity', 'theta_velocity', 'psi_velocity']
    a = data.iloc[[83]][cols].values.tolist()[0]
    b = data.iloc[[1]][cols].values.tolist()[0]

    print(hover_reward(np.array(a[0:3]),
                       np.array(a[3:6]),
                       np.array(a[6:9]),
                       np.array(a[9:12]),
                       np.array([0., 0., 10.])))

    print(hover_reward(np.array(b[0:3]),
                       np.array(b[3:6]),
                       np.array(b[6:9]),
                       np.array(b[9:12]),
                       np.array([0., 0., 10.])))

    #print(np.clip(5 * (1 - 2. * 98), -2, 3))

    # print(hover_reward(np.array([-1.729678297, -0.0012487256, 9.7895324748]),
    #                    np.array([6.2818577926, 3.6706919437, 3.5015792962]),
    #                    np.array([-7.0731331709, -0.003506869, -8.2229779699]),
    #                    np.array([-0.0053780827, -14.0411851482, -46.3242552138]),
    #                    np.array([0., 0., 10.])))
    # print(hover_reward(np.array([-2.1341031989, -0.0019159375, 9.2265339975]),
    #                    np.array([6.2815543693, 0.5227153292, 2.2772053046]),
    #                    np.array([-6.8387585351, -0.0125357532, -10.4270839967]),
    #                    np.array([-0.0044241086, -37.2056547725, -43.3475728102]),
    #                    np.array([0., 0., 10.])))


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
