import numpy as np
from matplotlib import pyplot as plt
import cv2, os
from pathlib import Path  # added

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# import pseudoslam.envs.simulator.util as util
# import pseudoslam.envs.simulator.jsonReader as jsonReader
#import misc.HouseExpo.pseudoslam.envs.simulator.util as util
#import misc.HouseExpo.pseudoslam.envs.simulator.jsonReader as jsonReader
from . import util
from . import jsonReader

import pickle
import lzma

import time

map_color= {'uncertain':-101, 'free':0, 'obstacle':100}
move_choice= {'forward':np.array([1,0]), 'left': np.array([0,1]), 'right': np.array([0,-1])}

class pseudoSlam():
    def __init__(self, param_file, obstacle_config=None):
        """ pseudoSlam initilization """
        np.random.seed(0)
        """ define class variable """
        self.m2p= 0
        self.robotRadius= 0
        self.stepLength_linear= 0
        self.stepLength_angular= 0
        self.config_poseInit= np.array([0,0,0])
        self.robotPose_init= np.array([0,0,0])
        self.robotResetRandomPose= 0
        self.obs_num= 0
        self.obs_sizeRange= np.array([0,0])
        self.laser_range=0
        self.laser_fov= 0
        self.laser_resol= 0
        self.laser_noiseSigma= 0
        self.slamErr_linear= 0
        self.slamErr_angular= 0
        self.state_size= np.array([0,0])
        self.world= np.zeros([1,1])
        self.obstacle_config = obstacle_config
        self.traj = []
        self.continues_actions = False # ADDED

        with open(param_file) as stream:
            self.config = yaml.load(stream, Loader=Loader)

        """json reader init"""
        self.json_reader = jsonReader.jsonReader(self.config['json_dir'], self.config['meter2pixel'])
        """ set map_color """
        self.map_color = map_color
        #self.map_id_set = np.loadtxt(os.path.join(os.path.dirname(__file__), "../", self.config['map_id_set']), str)  # Changed
        self.map_id_set = np.loadtxt(os.path.abspath(os.path.join(os.path.dirname(__file__), Path("../"), Path(self.config['map_id_set']))), str)  # Changed


        """ Initialize user config param """
        self.initialize_param(self.config)

        """ set motion choice """
        self.motionChoice= move_choice

        """ pre calculate radius and angle vector that will be used in building map """
        radius_vect= np.arange(self.laser_range+1)
        self.radius_vect= radius_vect.reshape(1, radius_vect.shape[0]) # generate radius vector of [0,1,2,...,laser_range]

        angles_vect = np.arange(-self.laser_fov*0.5, self.laser_fov*0.5,step=self.laser_resol)
        self.angles_vect = angles_vect.reshape(angles_vect.shape[0], 1) # generate angles vector from -laser_angle/2 to laser_angle

        self.robotPose = self.robotPose_init
        self.reset()

        return

    def initialize_param(self, config):
        """ world & robot param """
        self.m2p= config["meter2pixel"] # X pixel= 1 meter
        self.robotRadius= util.meter2pixel(config["robotRadius"], self.m2p) # robot radius in pixel
        self.stepLength_linear= util.meter2pixel(config["stepLength"]["linear"], self.m2p) # robot linear movement in each step in pixel
        self.stepLength_angular= util.deg2rad( config["stepLength"]["angular"] ) # robot angular movement in each step in rad

        """ robot starting pose """
        # robot starting pose in world coordinate and rad with the form of [y;x;theta]
        if isinstance(config["startPose"], str): # check if the startPose is contained in a file
            pickleFilePath = config["startPose"]
            f = lzma.open(pickleFilePath, 'rb')
            self.config_poseInit = pickle.load(f)
        else:
            self.config_poseInit[0]= config["startPose"]["y"]
            self.config_poseInit[1]= config["startPose"]["x"]
            self.config_poseInit[2]= util.deg2rad(config["startPose"]["theta"] )

        # flag of robot randomly reset start pose in each reset
        self.robotResetRandomPose= config["resetRandomPose"]


        """ obstacle param """
        self.obs_num= config["obstacle"]["number"] # number of obstacle added to the world
        # size of obstacle added in the form of [min;max] in pixel
        self.obs_sizeRange= np.zeros((2,1))
        self.obs_sizeRange[0]= util.meter2pixel( config["obstacle"]["size"]["min"], self.m2p)
        self.obs_sizeRange[1]= util.meter2pixel( config["obstacle"]["size"]["max"], self.m2p)

        """ laser param """
        self.laser_range= util.meter2pixel(config["laser"]["range"], self.m2p) # laser range in pixel
        self.laser_fov= util.deg2rad( config["laser"]["fov"] ) # laser field of view in rad
        self.laser_resol= util.deg2rad( config["laser"]["resolution"] ) # laser rotation resolution in rad
        self.laser_noiseSigma= config["laser"]["noiseSigma"] # sigma of Gaussian distribution of laser noise

        """ slam error """
        self.slamErr_linear= config["slamError"]["linear"] # slam linear registration error in pixel?
        self.slamErr_angular= util.deg2rad( config["slamError"]["angular"] ) # slam rotational registration error in rad

        """ state size """
        self.state_size= ( config["stateSize"]["x"] * self.m2p, config["stateSize"]["y"] * self.m2p ) # state size in the form of [x;y]

        """ unknown mode """
        self.is_exploration = (config["mode"] == 0)

        try: # ADDED
            self.continues_actions = config["continuesActions"]
        except:
            print("Assumes Discrete Actions")

        return

    def create_world(self, order=False, padding=10):
        """ read maps in order if True, else randomly sample"""


        if self.map_id_set.ndim > 0: # handle case with only one entry in map_id_set
            if order:
                map_id = self.map_id_set[0]
                self.map_id_set = np.delete(self.map_id_set, 0)
            else:
                LocalProcRandGen = np.random.RandomState()
                map_id = LocalProcRandGen.choice(self.map_id_set)
                # map_id = np.random.choice(self.map_id_set) <-- does not go well with multiprocessing
        else:
            map_id = np.str_(self.map_id_set)

        input_world, _ = self.json_reader.read_json(map_id)

        """ process world into simulator compatible map """
        self.world= self._map_process(input_world, padding=padding)

        (h,w)= self.world.shape

        if isinstance(self.config_poseInit, dict): # check if the startPose is contained in a file
            config_poseInit_ = np.array([0.0,0.0,0.0])
            config_poseInit_[0] = self.config_poseInit[map_id]["y"]
            config_poseInit_[1] = self.config_poseInit[map_id]["x"]
            config_poseInit_[2] = util.deg2rad(self.config_poseInit[map_id]["theta"])

        else:
            config_poseInit_ = self.config_poseInit

        self.robotPose_init[0:2]= util.world2mapCoord(config_poseInit_, (h*0.5,w*0.5), self.m2p)
        self.robotPose_init[2]= config_poseInit_[2]

        self.add_obstacle()
        self.map_id = map_id.copy()

        return self.world

    def _map_process(self,input_world, padding=5):
        """ process input map into simulator compatible map """

        map_gt = np.zeros_like(input_world)

        """ input world obstacle= 0 & free space= 255 | convert into simulator config """
        map_gt[input_world == 0] = self.map_color['obstacle']
        map_gt[input_world == 255] = self.map_color['free']

        """ crop out redundent obstacle region in boundaries """
        index = np.where(map_gt == self.map_color['obstacle'])
        [index_row_max, index_row_min, index_col_max, index_col_min] = \
            [np.max(index[0]), np.min(index[0]), np.max(index[1]), np.min(index[1])]
        map_gt = map_gt[index_row_min:index_row_max+1, index_col_min:index_col_max+1]

        """ pad the map with obstacle on near boundary """
        map_gt = np.lib.pad(map_gt, padding, mode='constant', constant_values=self.map_color['obstacle'])

        return map_gt

    def add_obstacle(self):
        """ Randomly add obstacle to world """
        if self.obstacle_config:
            """ add user defined obstacles """
            f = open(self.obstacle_config, "r")
            for config in f:
                x, y, w, h = [int(n) for n in config.split(" ")]
                rect = np.array([[x - w // 2, y - h // 2],
                                 [x - w // 2, y + h // 2],
                                 [x + w // 2, y + h // 2],
                                 [x + w // 2, y - h // 2]],
                                np.int32)
                cv2.fillPoly(self.world, [rect], self.map_color["obstacle"])
            return self.world

        if self.obs_num == 0:  # No obstacle added.
            return

        prox_min = 3 # min distance in pixel between added obstacle & obstacle in map
        world_obs = np.copy(self.world) # create a new image for adding obstacle
        (h,w)= self.world.shape

        for i in range(self.obs_num):
            while (1):
                # randomly generate obstacle orientation & obstacle size that fall within obstacle_sizeRange
                obs_a = np.random.randint(self.obs_sizeRange[0], self.obs_sizeRange[1])*0.5
                obs_b = np.random.randint(self.obs_sizeRange[0], self.obs_sizeRange[1])*0.5
                obs_theta = np.random.random()*360

                # check if obstacle length exceed 45% of world shape, if yes, randomly change obs_sizeRange to 5%~30% of min(h,w)
                obs_len= np.sqrt(obs_a*obs_a + obs_b*obs_b)*2
                if (obs_len*2>h*0.9 or obs_len*2>w*0.9):
                    self.obs_sizeRange[0]= np.random.randint(500,1500)*1.0/10000 *np.min((h,w))
                    self.obs_sizeRange[1]= np.random.randint(500,3000)*1.0/10000 *np.min((h,w))
                    self.obs_sizeRange= np.sort(self.obs_sizeRange,axis=0).astype(int)
                    print('obs_sizeRange exceed world shape, now changing to ',self.obs_sizeRange.tolist())
                    continue

                # randomly select shape type of obstacle [0: rectangle; 1: ellipse; 2: circle]
                obs_type = np.random.randint(0, 3)
                if obs_type == 2:
                    obs_b = obs_a

                # randomly generate obstacle center coordinate that the obstacle would not exceed world boundary
                bound = np.round(obs_len).astype(int) + prox_min
                obs_y = np.random.randint(bound, h-bound)
                obs_x = np.random.randint(bound, w-bound)

                # check if the location of obstacle to be added intersect with other obstacle present
                #if np.sum(world_obs[obs_y-bound:obs_y+bound, obs_x-bound:obs_x+bound]==self.map_color['obstacle'])!= 0:
                if np.sum(world_obs[obs_y[0]-bound[0]:obs_y[0]+bound[0], obs_x[0]-bound[0]:obs_x[0]+bound[0]]==self.map_color['obstacle'])!= 0:
                    continue

                print(world_obs)
                print((obs_x,obs_y))
                print((int(obs_a),int(obs_b)))
                print(obs_theta)

                # create obstacle patch
                if obs_type == 0:
                    cthe = np.cos(np.pi/180* obs_theta)
                    sthe = np.sin(np.pi/180* obs_theta)
                    rect = np.array([[obs_x + (-obs_a * cthe - -obs_b * sthe), obs_y + (-obs_a * sthe + -obs_b * cthe)],
                                     [obs_x + (-obs_a * cthe - obs_b * sthe), obs_y + (-obs_a * sthe + obs_b * cthe)],
                                     [obs_x + (obs_a * cthe - obs_b * sthe), obs_y + (obs_a * sthe + obs_b * cthe)],
                                     [obs_x + (obs_a * cthe - -obs_b * sthe), obs_y + (obs_a * sthe + -obs_b * cthe)]],
                                    np.int32)
                    cv2.fillPoly(world_obs, [rect], self.map_color["obstacle"])

                elif obs_type == 1:
                    # cv2.ellipse(world_obs, (obs_x,obs_y), (int(obs_a),int(obs_b)), obs_theta, 0, 360, self.map_color["obstacle"],thickness=-1)
                    cv2.ellipse(world_obs, (obs_x,obs_y), (int(obs_a),int(obs_b)), obs_theta, 0, 360, self.map_color["obstacle"],thickness=-1)
                else:
                    cv2.circle(world_obs, (obs_x,obs_y), int(obs_a), self.map_color["obstacle"],thickness=-1)

                break

        self.world= world_obs.copy()

        return world_obs

    def _randomizeRobotPose(self):
        # randomly generate robot start pose where robot is not crashed into obstacle
        h, w = self.world.shape
        x_min, x_max = int(0.1 * w), int(0.8 * w)
        y_min, y_max = int(0.1 * h), int(0.8 * h)

        # ################## CHANGED ##################
        LocalProcRandGen = np.random.RandomState()
        self.robotPose[0] = LocalProcRandGen.randint(y_min, y_max)
        self.robotPose[1] = LocalProcRandGen.randint(x_min, x_max) 
        # self.robotPose[0] = np.random.randint(y_min, y_max)  # <-- does not go well with multiprocessing
        # self.robotPose[1] = np.random.randint(x_min, x_max)  # <-- does not go well with multiprocessing

        while (self.robotCrashed(self.robotPose)):
            # ################## CHANGED ##################
            self.robotPose[0] = LocalProcRandGen.randint(y_min, y_max)
            self.robotPose[1] = LocalProcRandGen.randint(x_min, x_max) 
            # self.robotPose[0] = np.random.randint(y_min, y_max)  # <-- does not go well with multiprocessing
            # self.robotPose[1] = np.random.randint(x_min, x_max)  # <-- does not go well with multiprocessing
        self.robotPose[2] = np.random.rand() * np.pi * 2
        self.robotPose_init = self.robotPose.copy() # just to save it for "get_config_poseInit(...)"
        return self.robotPose

    def get_config_poseInit(self):
        (h,w)= self.world.shape
        config_poseInit = {}
        tmp = util.map2worldCoord(self.robotPose_init, (h*0.5,w*0.5), self.m2p)
        config_poseInit["y"] = tmp[0]
        config_poseInit["x"] = tmp[1]
        config_poseInit["theta"] = util.rad2deg(self.robotPose_init[2])

        return config_poseInit


    def reset(self, order=False):
        self.traj.clear()
        self.create_world(order)


        if (self.robotResetRandomPose==1) or (self.robotCrashed(self.robotPose_init)):
            # randomly generate robot start pose where robot is not crashed into obstacle
            self._randomizeRobotPose()
        else:
            self.robotPose = self.robotPose_init

        #self.config_poseInit= np.array([2,0,0])
        #(h,w)= self.world.shape
        #self.robotPose_init[0:2]= util.world2mapCoord(self.config_poseInit, (h*0.5,w*0.5), self.m2p)
        #self.robotPose_init[2]= self.config_poseInit[2]
        #self.robotPose = self.robotPose_init

        self.robotCrashed_flag= False
        if self.is_exploration:
            self.slamMap= np.ones(self.world.shape)*self.map_color["uncertain"]
            self.dslamMap= np.ones(self.world.shape)*self.map_color["uncertain"]
        else:
            self.slamMap = self.world.copy()
            self.dslamMap = self.world.copy()
        self.build_map(self.robotPose)
        return


    def _laser_noise(self, y_rangeCoordMat, x_rangeCoordMat, y_coord, x_coord, b):
        """ add laser noise, y_&x_coord are coord before obstacle | y_&x_rangeCoordMat are coord that the laser range covers
        b is the vector that represent the index of rangeCoordMat where the coord are before obstacle """

        noise_vector = np.random.normal(loc=0, scale=self.laser_noiseSigma, size=(x_rangeCoordMat.shape[0], 1))
        noise_vector = np.round(noise_vector).astype(np.int64)

        """ add noise to rangeCoordMat  """
        y_rangeCoordMat += noise_vector
        x_rangeCoordMat += noise_vector
        y_noise_coord = y_rangeCoordMat[b]
        x_noise_coord = x_rangeCoordMat[b]

        """ check for index of the coord that are within bound of world """
        inBound_ind= util.within_bound(np.array([y_noise_coord, x_noise_coord]),self.world.shape)

        """ get the coord that are within bound """
        x_noise_coord = x_noise_coord[inBound_ind]
        y_noise_coord = y_noise_coord[inBound_ind]
        x_coord = x_coord[inBound_ind]
        y_coord = y_coord[inBound_ind]

        return y_noise_coord, x_noise_coord, y_coord, x_coord

    def _slam_error(self, y_coord, x_coord):

        err_y = util.gauss_noise() * self.slamErr_linear
        err_x = util.gauss_noise() * self.slamErr_linear
        err_theta = util.gauss_noise() * self.slamErr_angular

        """ rotate y_coord & x_coord by err_theta """
        (y_coord_err, x_coord_err)= util.transform_coord(y_coord,x_coord, self.robotPose, np.array([err_y,err_x,err_theta]))

        """ check for index where the coord are within bound """
        inBound_ind= util.within_bound(np.array([y_coord_err,x_coord_err]),self.world.shape)
        x_err_coord= x_coord_err[inBound_ind]
        y_err_coord= y_coord_err[inBound_ind]

        x_coord = x_coord.reshape(x_coord.shape[0], 1)[inBound_ind]
        y_coord = y_coord.reshape(y_coord.shape[0], 1)[inBound_ind]

        return y_err_coord, x_err_coord, y_coord, x_coord

    def _laser_slam_error(self, y_rangeCoordMat, x_rangeCoordMat, y_coord, x_coord, b):
        [y_noise_rangeCoordMat, x_noise_rangeCoordMat, y_coord, x_coord] = self._laser_noise(y_rangeCoordMat, x_rangeCoordMat, y_coord, x_coord, b)
        [y_err_coord, x_err_coord, y_noise_ind, x_noise_ind] = self._slam_error(y_noise_rangeCoordMat, x_noise_rangeCoordMat)

        inBound_ind= util.within_bound(np.array([y_err_coord,x_err_coord]),self.world.shape)
        y_coord = y_coord[inBound_ind]
        x_coord = x_coord[inBound_ind]
        return y_err_coord, x_err_coord, y_coord, x_coord


    def _build_map_with_rangeCoordMat(self, y_rangeCoordMat, x_rangeCoordMat):
        # Round y and x coord into int
        y_rangeCoordMat = (np.round(y_rangeCoordMat)).astype(np.int)
        x_rangeCoordMat = (np.round(x_rangeCoordMat)).astype(np.int)

        """ Check for index of y_mat and x_mat that are within the world """
        inBound_ind= util.within_bound(np.array([y_rangeCoordMat, x_rangeCoordMat]), self.world.shape)

        """ delete coordinate that are not within the bound """
        outside_ind = np.argmax(~inBound_ind, axis=1)
        ok_ind = np.where(outside_ind == 0)[0]
        need_amend_ind = np.where(outside_ind != 0)[0]
        outside_ind = np.delete(outside_ind, ok_ind)

        inside_ind = np.copy(outside_ind)
        inside_ind[inside_ind != 0] -= 1
        bound_ele_x = x_rangeCoordMat[need_amend_ind, inside_ind]
        bound_ele_y = y_rangeCoordMat[need_amend_ind, inside_ind]

        count = 0
        for i in need_amend_ind:
            x_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_x[count]
            y_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_y[count]
            count += 1

        """ find obstacle along the laser range """
        obstacle_ind = np.argmax(self.world[y_rangeCoordMat, x_rangeCoordMat] == self.map_color['obstacle'], axis=1)
        obstacle_ind[obstacle_ind == 0] = x_rangeCoordMat.shape[1]


        """ generate a matrix of [[1,2,3,...],[1,2,3...],[1,2,3,...],...] for comparing with the obstacle coord """
        bx = np.arange(x_rangeCoordMat.shape[1]).reshape(1, x_rangeCoordMat.shape[1])
        by = np.ones((x_rangeCoordMat.shape[0], 1))
        b = np.matmul(by, bx)

        """ get the coord that the robot can percieve (ignore pixel beyond obstacle) """
        b = b <= obstacle_ind.reshape(obstacle_ind.shape[0], 1)
        y_coord = y_rangeCoordMat[b]
        x_coord = x_rangeCoordMat[b]


        """ no slam error """
        # self.slamMap[y_coord, x_coord] = self.world[y_coord, x_coord]

        """ laser noise """
        # [y_noise_ind,x_noise_ind, y_coord,x_coord]= self._laser_noise(y_rangeCoordMat,x_rangeCoordMat,y_coord,x_coord,b)
        # self.slamMap[y_noise_ind,x_noise_ind]= self.world[y_coord,x_coord]
        #

        """ slam matching error """
        # [y_err_ind,x_err_ind, y_coord,x_coord] = self._slam_error(y_coord,x_coord)
        # self.slamMap[y_err_ind,x_err_ind]= self.world[y_coord,x_coord]

        """ laser noise + slam matching error """
        [y_all_noise,x_all_noise, y_coord,x_coord]= self._laser_slam_error(y_rangeCoordMat,x_rangeCoordMat,y_coord,x_coord,b)
        self.slamMap[y_all_noise,x_all_noise]= self.world[y_coord,x_coord]

        self.y_all_noise = y_all_noise 
        self.x_all_noise = x_all_noise

        """ dilate/close to fill the holes """
        # self.dslamMap= cv2.morphologyEx(self.slamMap,cv2.MORPH_CLOSE,np.ones((3,3)))
        self.dslamMap= cv2.dilate(self.slamMap, np.ones((3,3)), iterations=1)
        return self.slamMap

    def build_map(self, pose):
        """ build perceived map based on robot position and its simulated laser info
        pose: [y;x;theta] in pixel in img coord | robotPose= pose"""
        """ input pose can be in decimal place, it will be rounded off in _build_map_with_rangeCoordMat """

        self.robotPose= pose
        """ find the coord matrix that the laser cover """
        angles= pose[2] + self.angles_vect
        y_rangeCoordMat= pose[0] - np.matmul(np.sin(angles), self.radius_vect)
        x_rangeCoordMat= pose[1] + np.matmul(np.cos(angles), self.radius_vect)

        self._build_map_with_rangeCoordMat(y_rangeCoordMat,x_rangeCoordMat)
        return self.slamMap

    def moveRobot(self, moveAction): ################### ADDED!! ###################
        if self.continues_actions:
            return self.moveRobot_continues(moveAction)
        else:
            return self.moveRobot_discrete(moveAction)

    def moveRobot_continues(self, moveAction):
        motion = moveAction
        dvX = motion[0]*self.stepLength_linear # x motion
        dvY = motion[1]*self.stepLength_linear # y motion
        dtheta = motion[2]*self.stepLength_angular # rotational motion
        """ oversample the motion in each step """
        # sampleNo= 2
        samplePixel= 7
        sampleNo= 50#np.max([np.abs(dvX)*1.0/samplePixel, np.abs(dvX)*1.0/samplePixel]) ## consider something that is not hard coded!

        moveLength_step= dvX*1./sampleNo
        moveLength_total= 0
        i=0
        while(i<sampleNo):

            # check if remaining step < moveLength_step, if yes, just move the remaining length instead of the whole moveLength_step
            remain_length= self.stepLength_linear-moveLength_total
            if 0< remain_length and remain_length <moveLength_step:
                moveLength_step= self.stepLength_linear-moveLength_total

            y= self.robotPose[0] + dvX/sampleNo
            x= self.robotPose[1] + dvY/sampleNo
            # theta= self.robotPose[2] # stays the same
            theta= self.robotPose[2] + dtheta/sampleNo

            targetPose= np.array([y,x,theta])

            # check if robot will crash on obstacle or go out of bound
            if self.robotCrashed(targetPose):
                self.robotCrashed_flag= True
                # print("Robot crash")
                return False

            # if moveAction == "forward":
            #    self.traj.append([int(x), int(y)])  # only save distince pts

            # build map on the targetPose
            self.build_map( targetPose )
            i=i+1
            moveLength_total+= samplePixel

        return True

    def moveRobot_discrete(self, moveAction):  # NAME HAS CHANGED!
        """ move robot with moveAction with forward | left | right """
        motion= self.motionChoice[moveAction]
        dv= motion[0]*self.stepLength_linear # forward motion
        dtheta= motion[1]*self.stepLength_angular # angular motion

        """ oversample the motion in each step """
        # sampleNo= 2
        samplePixel= 7
        sampleRad= np.pi/180*10
        sampleNo= np.max([np.abs(dv)*1.0/samplePixel, np.abs(dtheta)*1.0/sampleRad])

        moveLength_step= dv*1./sampleNo
        moveLength_total= 0
        i=0
        while(i<sampleNo):
            # print(self.robotPose)
            theta= self.robotPose[2] + dtheta*1.0/sampleNo
            theta= np.arctan2(np.sin(theta),np.cos(theta))

            # check if remaining step < moveLength_step, if yes, just move the remaining length instead of the whole moveLength_step
            remain_length= self.stepLength_linear-moveLength_total
            if 0< remain_length and remain_length <moveLength_step:
                moveLength_step= self.stepLength_linear-moveLength_total

            y= self.robotPose[0] - np.sin(theta)*moveLength_step
            x= self.robotPose[1] + np.cos(theta)*moveLength_step
            targetPose= np.array([y,x,theta])

            # check if robot will crash on obstacle or go out of bound
            if self.robotCrashed(targetPose):
                self.robotCrashed_flag= True
                # print("Robot crash")
                return False

            if moveAction == "forward":
                self.traj.append([int(x), int(y)])  # only save distince pts

            # build map on the targetPose
            self.build_map( targetPose )
            i=i+1
            moveLength_total+= samplePixel

        return True

    def world2state(self):
        # state= cv2.resize(self.slamMap, self.state_size, interpolation=cv2.INTER_LINEAR)
        state= self.slamMap.copy()
        # draw robot position on state
        cv2.circle(state, (int(self.robotPose[1]), int(self.robotPose[0])), self.robotRadius, 50, thickness=-1)

        # draw robot orientation heading on state
        headRadius = np.ceil(self.robotRadius/3.).astype(np.int)
        headLen = self.robotRadius + headRadius
        # orientPt = util.transform_coord(self.robotPose[0], self.robotPose[1], self.robotPose, np.array([0, headLen, 0]))
        # cv2.circle(state, (orientPt[1],orientPt[2]), headRadius, 50, thickness=-1)
        head_y = self.robotPose[0] - np.sin(self.robotPose[2]) * headLen
        head_x = self.robotPose[1] + np.cos(self.robotPose[2]) * headLen
        cv2.circle(state, (int(head_x), int(head_y)), headRadius, 50, thickness=-1)

        if not self.is_exploration:
            """Change color for known environment navigation"""
            state[state == self.map_color['free']] = 255
            state[state == self.map_color['obstacle']] = 0
        return state

    def robotCrashed(self, pose):
        if ~util.within_bound(pose, self.world.shape, self.robotRadius):
            return True

        py= np.round(pose[0]).astype(int)
        px= np.round(pose[1]).astype(int)
        r= self.robotRadius

        # make a circle patch around robot location and check if there is obstacle pixel inside the circle
        robotPatch, _ = util.make_circle(r,1)
        worldPatch= self.world[py-r:py+r+1, px-r:px+r+1]
        worldPatch= worldPatch*robotPatch

        return np.sum(worldPatch==self.map_color["obstacle"])!=0


    def get_state(self):
        return self.world2state().copy()
        # return self.slamMap.copy()

    def get_pose(self):
        return self.robotPose.copy()

    def get_crashed(self):
        return self.robotCrashed_flag

    def measure_ratio(self):  # exploration percentage
        world_pixel = np.sum(self.world==self.map_color['free'])

        ################### CHANGED!! ###################
        # This way of calculating it has the problem that when noise is used the ratio can be greater than 1
        # i.e. we have explored more of the map than possible
        # mapped_pixel= np.sum(self.slamMap==self.map_color['free'])
        # world_pixel= np.sum(self.world==self.map_color['free'])

        # We should only count the pixels that actually can be observed, but it is hard
        # to distinguish which of the occupied pixels can actually be observed (only the boarder)
        # so we only count the free space pixels.
        # Here we do not distinguish between free and occupied pixels, since we believ that as long
        # as the value of the pixels is not perfectly "uncertain" it should be considered explored
        # eventhough the current map estimate is wrong!
        true_free_idx = self.world==self.map_color['free']
        mapped_pixel= np.sum(self.slamMap[true_free_idx]!=self.map_color['uncertain'])

        return 1.*mapped_pixel/world_pixel

