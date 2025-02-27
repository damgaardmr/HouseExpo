import numpy as np
import cv2
from os import path
from matplotlib import pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding

from pathlib import Path
if __name__ == '__main__':
    from simulator.pseudoSlam import pseudoSlam
else:
    from .simulator.pseudoSlam import pseudoSlam

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

class RobotExplorationProbMind(gym.Env):
    def __init__(self, config_path='config_probMind.yaml'):
        if config_path.startswith("/"):
            fullpath = config_path
        else:
            fullpath = path.join(path.dirname(__file__), "config", config_path)
        print(fullpath)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        with open(Path(fullpath), 'r') as stream:
            try:
                configs = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        print("fullpath: " + str(fullpath))
        self.sim = pseudoSlam(Path(fullpath))
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.last_map = self.sim.get_state()
        self.last_action = None

        self.configs = configs
        self.goal_always_observed = configs["goal_always_observed"]
        if self.configs["reachGoalMode"]:
            if "randomGoal" in self.configs:
                self.randomGoal = self.configs["randomGoal"]
                self.goal_position_config = [self.configs["goalPosition"]["x"], self.configs["goalPosition"]["y"]]
            else:
               self.randomGoal = True
            self.generate_goal()


    def generate_goal(self,recursion_depth=0):
        goal_radius = self.configs["goal_zone_radius"]
        r_in_pixels = int(np.ceil(goal_radius * self.configs["meter2pixel"]))
        color = (255, 255, 255)
        thickness = -1  # fully filled circle

        reachable = False
        while not reachable:  # check if the goal zone collide with any obstacles in the world

            if self.randomGoal:
                print("r_in_pixels: " + str(r_in_pixels))
                print("sim.world.shape: " + str(self.sim.world.shape))
                print("sim.world.shape: " + str(self.sim.world.shape))
                g_x = np.random.randint(r_in_pixels, self.sim.world.shape[1] - r_in_pixels, (1,))[0]
                g_y = np.random.randint(r_in_pixels, self.sim.world.shape[0] - r_in_pixels, (1,))[0]
            else:
                g_x = int(self.goal_position_config[0]*self.configs["meter2pixel"])
                g_y = int(self.goal_position_config[1]*self.configs["meter2pixel"])

            goal_pos_in_pixels = [g_x, g_y]
            world_with_goal_zone = self.sim.world.copy()
            cv2.circle(world_with_goal_zone, goal_pos_in_pixels, r_in_pixels, color, thickness)
            goal_zone_mask = self.sim.world!=world_with_goal_zone
            if np.sum(np.abs(self.sim.world[goal_zone_mask])) == 0:
                reachable = True
            else:
                if recursion_depth == 100 and not self.randomGoal:
                    raise NameError('Invalid Goal zone location! Failed to generate goal after ' + str(recursion_depth) + ' tries')
                else:
                    self.generate_goal(recursion_depth=recursion_depth+1)

        self.goal_zone_mask = goal_zone_mask
        self.goal_pos_in_pixels = goal_pos_in_pixels
        self.goal_pos_in_m = [g_x / self.configs["meter2pixel"], g_y / self.configs["meter2pixel"]]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        slamMap = self._get_obs()[:, :, 0]
        if mode == "human":
            plt.figure(0)
            plt.clf()
            plt.imshow(slamMap, cmap='gray')
            plt.draw()
            plt.pause(0.00001)
        elif mode == "rgb_array":
            obs = np.expand_dims(slamMap, -1).repeat(3, -1)
            obs += 101  # convert to [0,255]
            obs = obs.astype(np.uint8)
            return obs

    def reset(self, order=False):
        self.sim.reset(order)
        self.last_map = self.sim.get_state()
        self.last_action = None
        if self.configs["reachGoalMode"]:
            self.generate_goal()
        return self._get_obs()

    def step(self, action):
        action_ = np.array([action[1], action[0], action[2]])  # change from [x,y] to [y,x] convention

        crush_flag = self.sim.moveRobot(action_)

        pose_pixels = self.sim.get_pose()  # + noise on pose?????
        self.pose = np.array([pose_pixels[1] / self.sim.m2p, pose_pixels[0] / self.sim.m2p, pose_pixels[2]])  # change from [y,x] to [x,y] convention and in meters instead of pixels

        grid_map_with_robot = self.sim.get_state().copy()  # returns map with robot on top
        grid_map_ = self.sim.slamMap.copy()  # only returns map
        # the get_state returns:
        # - 101 for unknown
        # 0 for unoccupied
        # 101 occupied
        # we need:
        # 0 for unoccupied
        # 0.5 for unknown
        # 1 occupied
        # normalize
        unknowns = np.where(grid_map_ == -101)  # find id of unknowns
        grid_map_[unknowns] = 50.0  # find id of unknowns
        grid_map = grid_map_ / (100 - 0)

        if self.configs["reachGoalMode"]:
            goal = self._check_if_goal_is_visable()
            done = self._check_if_robot_has_reach_goal_zone()
        else:
            done = (self.sim.measure_ratio() > 0.95)
            goal = None
        obs = [self.pose, grid_map, goal]

        reward = self._compute_reward(crush_flag, action)
        info = {'is_success': done, 'is_crashed': self.sim.robotCrashed_flag, 'grid_map_with_robot': grid_map_with_robot}

        if self.sim.robotCrashed_flag:  # reset crash flag
            self.sim.robotCrashed_flag = False

        return obs, reward, done, info

    def close(self):
        pass

    def _check_if_robot_has_reach_goal_zone(self):
        dist_vector = np.array([self.pose[0] - self.goal_pos_in_m[0],
                                self.pose[1] - self.goal_pos_in_m[1]])
        goal_dist = np.linalg.norm(dist_vector)

        # print([self.pose[0], self.pose[1]])
        # print(self.goal_pos_in_m)
        # print(dist_vector)
        # print(goal_dist)
        # print(goal_dist <= self.configs["goal_zone_radius"])

        if goal_dist <= self.configs["goal_zone_radius"]:
            return True
        else:
            return False

    def _check_if_goal_is_visable(self):
        if self.goal_always_observed:
            return self.goal_pos_in_m
        else:
            visible_slamMap = np.zeros_like(self.sim.slamMap, dtype=bool)
            visible_slamMap[self.sim.y_all_noise,self.sim.x_all_noise] = True

            where_equal_mask = np.logical_and(self.goal_zone_mask, visible_slamMap)
            where_equal_idx = np.argwhere(where_equal_mask)

            # fig = plt.figure(101)
            # plt.imshow(self.goal_zone_mask) #, cmap='gray')
            # fig.canvas.draw()
            # fig.canvas.flush_events()

            # fig = plt.figure(104)
            # plt.imshow(visible_slamMap) #, cmap='gray')
            # fig.canvas.draw()
            # fig.canvas.flush_events()

            # fig = plt.figure(105) #(np.random.randint(1000))
            # plt.imshow(where_equal_mask) #, cmap='gray')
            # fig.canvas.draw()
            # fig.canvas.flush_events()


            if np.sum(where_equal_idx)>0:
                # draw random index as the center of the goal zone
                # idx = np.random.randint(where_equal_idx.shape[0])
                # noisy_pos = [where_equal_idx[idx][1]/self.configs["meter2pixel"], where_equal_idx[idx][0]/self.configs["meter2pixel"]]

                # calculate centroid of visible pixels
                centroid = np.mean(where_equal_idx, axis=0)
                pos = [centroid[1]/self.configs["meter2pixel"], centroid[0]/self.configs["meter2pixel"]]
                cov = np.eye(2) * self.configs["goal_zone_est_error_3_sigma"] / 3  # 99.7% of samples within a circle of "initial_3_sigma" cm
                noisy_pos = np.random.multivariate_normal(pos, cov)

                return noisy_pos
            else:
               return None  # the goal is not visable

    def _compute_reward(self, crush_flag, action):
        """Recurn the reward"""
        current_map = self.sim.get_state()
        difference_map = np.sum(self.last_map == self.sim.map_color['uncertain'])\
            - np.sum(current_map == self.sim.map_color['uncertain'])
        self.last_map = current_map
        # exploration
        reward = (1. * difference_map / self.sim.m2p / self.sim.m2p)

        return reward

    def _get_action_space(self):
        """Forward, left and right"""
        # return spaces.Discrete(3)
        return spaces.Box(np.float32(np.array([-1, -1, -1])), np.float32(np.array([1, 1, 1])), dtype='float32')

    def _get_observation_space(self):
        obs = self._get_obs()
        observation_space = spaces.Box(np.float32(-np.inf), np.float32(np.inf), shape=obs.shape, dtype='float32')
        return observation_space

    def _get_obs(self):
        """
        """
        observation = self.sim.get_state()
        pose = self.sim.get_pose()

        (rot_y, rot_x) = (int(pose[0]), int(pose[1]))
        rot_theta = -pose[2] * 180. / np.pi + 90  # Upward

        # Pad boundaries
        pad_x, pad_y = int(self.sim.state_size[0] / 2. * 1.5), int(self.sim.state_size[1] / 2. * 1.5)
        state_size_x, state_size_y = int(self.sim.state_size[0]), int(self.sim.state_size[1])
        if rot_y - pad_y < 0:
            observation = cv2.copyMakeBorder(observation, top=pad_y, bottom=0, left=0, right=0,
                                             borderType=cv2.BORDER_CONSTANT, value=self.sim.map_color['uncertain'])
            rot_y += pad_y
        if rot_x - pad_x < 0:
            observation = cv2.copyMakeBorder(observation, top=0, bottom=0, left=pad_x, right=0,
                                             borderType=cv2.BORDER_CONSTANT, value=self.sim.map_color['uncertain'])
            rot_x += pad_x
        if rot_y + pad_y > observation.shape[0]:
            observation = cv2.copyMakeBorder(observation, top=0, bottom=pad_y, left=0, right=0,
                                             borderType=cv2.BORDER_CONSTANT, value=self.sim.map_color['uncertain'])
        if rot_x + pad_x > observation.shape[1]:
            observation = cv2.copyMakeBorder(observation, top=0, bottom=0, left=0, right=pad_x,
                                             borderType=cv2.BORDER_CONSTANT, value=self.sim.map_color['uncertain'])

        # Rotate global map and crop the local observation
        local_map = observation[rot_y - pad_y:rot_y + pad_y, rot_x - pad_x:rot_x + pad_x]
        M = cv2.getRotationMatrix2D((pad_y, pad_x), rot_theta, 1)
        dst = cv2.warpAffine(local_map, M, (pad_y * 2, pad_x * 2), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=self.sim.map_color['uncertain'])
        dst = dst[pad_y - int(state_size_y / 2.):pad_y + int(state_size_y / 2.),
                  pad_x - int(state_size_x / 2.):pad_x + int(state_size_x / 2.)]
        dst = dst[:, :, np.newaxis]

        # Draw the robot at the center
        cv2.circle(dst, (int(state_size_y / 2.), int(state_size_x / 2.)), int(self.sim.robotRadius), 50, thickness=-1)
        cv2.rectangle(dst, (int(state_size_y / 2.) - int(self.sim.robotRadius),
                            int(state_size_x / 2.) - int(self.sim.robotRadius)),
                      (int(state_size_y / 2.) + int(self.sim.robotRadius),
                       int(state_size_x / 2.) + int(self.sim.robotRadius)),
                      50, -1)
        return dst.copy()


if __name__ == '__main__':
    env = RobotExplorationProbMind()
    env.reset()

    epi_cnt = 0
    while 1:
        pose = env.sim.get_pose()
        plt.figure(1)
        plt.clf()
        plt.imshow(env.sim.get_state().copy(), cmap='gray')
        plt.draw()
        plt.pause(0.1)
        env.render()

        epi_cnt += 1
        act = np.random.rand(3) * 2 - 1
        obs, reward, done, info = env.step(act)

        if epi_cnt > 100 or done:
            epi_cnt = 0
            env.reset()
