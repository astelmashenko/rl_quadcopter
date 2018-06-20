import numpy as np
from physics_sim import PhysicsSim
from utils import eucl_distance, udacity_distance, hover_reward


class Task:
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 50  # 0
        self.action_high = 1500  # 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        #
        self.dist = eucl_distance(self.sim.pose[:3], self.target_pos)

    def distance(self):
        return udacity_distance(self.sim.pose[:3], self.target_pos)

    def eucl_distance(self):
        return eucl_distance(self.sim.pose[:3], self.target_pos)

    def check_boundaries(self):
        boundaries = False
        for ii in range(3):
            if (self.sim.pose[ii] <= self.sim.lower_bounds[ii]) or (self.sim.pose[ii] >= self.sim.upper_bounds[ii]):
                boundaries = True
        return boundaries

    def get_reward2(self):
        """Uses current pose of sim to return reward."""
        dist = self.distance()
        eucl = self.eucl_distance()

        reward = -np.log(eucl ** 10 + 0.000000000001)

        reward -= 0.25 * udacity_distance(self.sim.pose[:2], self.target_pos[:2])
        reward -= 0.5 * udacity_distance(self.sim.pose[2], self.target_pos[2])

        # reward for velocity along z axe
        reward += self.sim.v[2]
        # if self.sim.v[2] > 0:
        #     reward += -np.log(self.sim.v[2] + 0.001)
        # elif self.sim.v[2] < 0:
        #     reward -= -np.log(abs(self.sim.v[2]) + 0.001)

        # penalize for angle
        reward -= 5 * abs(self.sim.pose[3:] / (3 * 2 * np.pi)).sum()
        return reward

    def get_reward(self):
        if self.check_boundaries():
            return -100
        return hover_reward(self.sim.pose[:3], self.sim.pose[3:], self.sim.v, self.sim.angular_v, self.target_pos)

    def get_reward1(self):
        reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        return np.clip(reward, -1, 1)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        # eucl = self.eucl_distance()

        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            # # if we close enough to target position end
            # if eucl <= 0.1:
            #     reward += 100
            #     done = True

        self.dist = self.eucl_distance()
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        self.dist = eucl_distance(self.sim.pose[:3], self.target_pos)
        return state
