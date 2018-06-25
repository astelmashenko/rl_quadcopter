import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pow, log
from sklearn.model_selection import ParameterSampler


def udacity_distance(x, y):
    return (abs(x - y)).sum()


def eucl_distance(x, y):
    return np.linalg.norm(x - y)


def remap(x, in_min, in_max, out_min=-1, out_max=1):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def eucl_distance_reward(x):
    return np.clip(weight_fun(6, 2.4, log(x)), -15, 15)


def z_diff_reward(x):
    return np.clip(weight_fun(4, 2., log(x)), -10, 10)


def x_diff_reward(x):
    return np.clip(weight_fun(4, 2., log(x)), -5, 5)


def y_diff_reward(x):
    return np.clip(weight_fun(4, 4., log(x)), -5, 5)


def euler_angles_reward(x):
    return np.clip(weight_fun(-2, 3., log(x)), -4, 4)


def velocities_reward(x):
    return np.clip(weight_fun(2, 1., log(abs(x-3))), -2, 2)


def weight_fun(a, b, x):
    return a - b * x


def hover_reward(pose, ang_pose, v, ang_v, target_pose):
    x = abs(pose[0] - target_pose[0]).sum()
    y = abs(pose[1] - target_pose[1]).sum()
    z = abs(pose[2] - target_pose[2]).sum()
    z_v = abs(v[2])
    euler_angles = abs(ang_pose / (3 * 2 * np.pi)).sum()

    reward = 0
    # np.clip(, -1, 1)
    eucl_dist = eucl_distance(pose, target_pose)
    reward += eucl_distance_reward(eucl_dist)

    # z distance
    reward += z_diff_reward(z)
    # # xy distance
    # # reward += np.clip(5 * (1 - 2. * xy_reward), -4, 3)
    reward += x_diff_reward(x)
    reward += y_diff_reward(y)
    #
    # # angles
    reward += euler_angles_reward(euler_angles)

    reward += velocities_reward(z_v)

    # reward += np.clip(weight_fun(1, 1, ang_xyz_v), -1, 1)
    # return reward
    return remap(reward, -41, 41, -1, 1)



def eucl_dist_test():
    print('eucl dist test')
    eucl_dists = [150, 100, 75, 50, 25, 10, 5, 1.1, 0.1]
    for dist in eucl_dists:
        print('%s\t:\t%s' % (dist, eucl_distance_reward(dist)))


def z_axe_test():
    print('z dist test')
    zs = [0.1, 0.5, 1.1, 2, 5, 10, 50]
    for z in zs:
        print('%s\t:\t%s' % (z, z_diff_reward(z)))


def x_axe_test():
    print('x dist test')
    xs = [0.1, 0.5, 1.1, 2, 5, 10, 50]
    for x in xs:
        print('%s\t:\t%s' % (x, x_diff_reward(x)))


def y_axe_test():
    print('y dist test')
    ys = [0.1, 0.5, 1.1, 2, 5, 10, 50]
    for y in ys:
        print('%s\t:\t%s' % (y, y_diff_reward(y)))


def euler_angles_test():
    print('euler angles test')
    angles = [0.1, 0.3, 0.4, 0.5, 1.1, 2, np.pi, 5, 2*np.pi]
    for angle in angles:
        print('%s\t:\t%s' % (angle, euler_angles_reward(angle)))


def velocities_reward_test():
    print('velocities angles test')
    vs = [-20, -10, -5, -1, 0.1, 1, 2, 5, 10, 15, 20]
    for v in vs:
        print('%s\t:\t%s' % (v, velocities_reward(v)))



if __name__ == '__main__':
    # data = pd.read_csv('data1.csv')
    # cols = ['x', 'y', 'z',
    #         'phi', 'theta', 'psi',
    #         'x_velocity', 'y_velocity', 'z_velocity',
    #         'phi_velocity', 'theta_velocity', 'psi_velocity']
    # a = data.iloc[[10]][cols].values.tolist()[0]
    # b = data.iloc[[40]][cols].values.tolist()[0]
    #
    # print(hover_reward(np.array(a[0:3]),
    #                    np.array(a[3:6]),
    #                    np.array(a[6:9]),
    #                    np.array(a[9:12]),
    #                    np.array([0., 0., 10.])))
    #
    # print(hover_reward(np.array(b[0:3]),
    #                    np.array(b[3:6]),
    #                    np.array(b[6:9]),
    #                    np.array(b[9:12]),
    #                    np.array([0., 0., 10.])))

    eucl_dist_test()
    z_axe_test()
    x_axe_test()
    y_axe_test()
    euler_angles_test()
    velocities_reward_test()

    # pose, ang, v, ang_v
    target = np.array([0., 0., 50.])

    param_grid = {'': [0.007, 0.01, 0.015],
                  'alpha': [0.15, 0.2, 0.25],
                  'gamma': [1.0, 0.9],
                  'q_update': ['sarsamax', 'exp_sarsa0']}

    print(hover_reward(np.array([-1., 0.1, 9.]),
                       np.array([6., 3., 3.]),
                       np.array([-7., 0.1, -8.]),
                       np.array([-0., -14., -46.]),
                       np.array([0., 0., 10.])))

    print(hover_reward(np.array([-2., -10., 9.]),
                       np.array([6., 0., 2.]),
                       np.array([-6., 0., -10.]),
                       np.array([-0., -37., -43.]),
                       np.array([0., 0., 10.])))

    print(hover_reward(np.array([-2., -100., 9.]),
                       np.array([0.1, 0.3, 0.2]),
                       np.array([-6., 0., -10.]),
                       np.array([-0., -37., -43.]),
                       np.array([0., 0., 10.])))


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
