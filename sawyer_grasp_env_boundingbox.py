"""
The environment of Sawyer Arm + Baxter Gripper for graping object.
With a bounding box of the arange that the gripper cannot move outside.
"""
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.sawyer import Sawyer
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.const import JointType, JointMode
import numpy as np
import matplotlib.pyplot as plt
import math

POS_MIN, POS_MAX = [0.1, -0.3, 1.], [0.45, 0.3, 1.]  # valid position range of target object 


class GraspEnv(object):
    ''' Sawyer robot grasping a cuboid '''
    def __init__(self, headless, control_mode='joint_velocity'):
        '''
        parameters:
        :headless: bool, if True, no visualization, else with visualization.
        :control mode: str, 'end_position' or 'joint_velocity'.
        '''
        # set public variables
        self.headless = headless   # if headless is True, no visualization
        self.reward_offset = 10.0  # reward of achieving the grasping object
        self.reward_range = self.reward_offset # reward range for register gym env when using vectorized env wrapper
        self.penalty_offset = 1.  # penalty value for undesired cases
        self.fall_down_offset = 0.1 # distance for judging the target object fall off the table
        self.metadata=[]  # gym env argument
        self.control_mode = control_mode  # the control mode of robotic arm: 'end_position' or 'joint_velocity'

        # launch and set up the scene, and set the proxy variables in represent of the counterparts in the scene
        self.pr = PyRep()   # call the PyRep
        if control_mode == 'end_position':  # need to use different scene, the one with all joints in inverse kinematics mode
            SCENE_FILE = join(dirname(abspath(__file__)), './scenes/sawyer_reacher_rl_new_ik.ttt')  # scene with joints controlled by ik (inverse kinematics)
        elif control_mode == 'joint_velocity': # the scene with all joints in force/torch mode for forward kinematics
            SCENE_FILE = join(dirname(abspath(__file__)), './scenes/sawyer_reacher_rl_new.ttt')  # scene with joints controlled by forward kinematics
        self.pr.launch(SCENE_FILE, headless=headless)  # lunch the scene, headless means no visualization
        self.pr.start()       # start the scene
        self.agent = Sawyer()  # get the robot arm in the scene
        self.gripper = BaxterGripper()  # get the gripper in the scene
        self.gripper_left_pad = Shape('BaxterGripper_leftPad')  # the left pad on the gripper finger
        self.proximity_sensor = ProximitySensor('BaxterGripper_attachProxSensor')  # need the name of the sensor here
        self.vision_sensor = VisionSensor('Vision_sensor')  # need the name of the sensor here
        self.table  = Shape('diningTable')  # the table in the scene for checking collision
        if control_mode == 'end_position':  # control the robot arm by the position of its end using inverse kinematics
            self.agent.set_control_loop_enabled(True)  # if false, inverse kinematics won't work
            self.action_space = np.zeros(4)  # 3 DOF end position control + 1 rotation of gripper
        elif control_mode == 'joint_velocity':  # control the robot arm by directly setting velocity values on each joint, using forward kinematics
            self.agent.set_control_loop_enabled(False)
            self.action_space = np.zeros(7)  # 7 DOF velocity control, no need for extra control of end rotation, the 7th joint controls it.
        else:
            raise NotImplementedError
        self.observation_space = np.zeros(17)  # position and velocity of 7 joints + position of the target
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')  # get the target object
        self.agent_ee_tip = self.agent.get_tip()  # a part of robot as the end of inverse kinematics chain for controlling
        self.tip_target = Dummy('Sawyer_target')   # the target point of the tip (end of the robot arm) to move towards
        self.tip_pos = self.agent_ee_tip.get_position()  # tip x,y,z position
        
        # set a proper initial robot gesture or tip position
        if control_mode == 'end_position': 
            initial_pos = [0.3, 0.1, 0.9]
            self.tip_target.set_position(initial_pos)
            # one big step for rotation setting is enough, with reset_dynamics=True, set the rotation instantaneously
            self.tip_target.set_orientation([0,np.pi,np.pi/2], reset_dynamics=True)  # first two dimensions along x and y axis make gripper face downwards  
        elif control_mode == 'joint_velocity':
            self.initial_joint_positions = [0.001815199851989746, -1.4224984645843506, \
                0.704303503036499, 2.54307222366333, 2.972468852996826, -0.4989511966705322, 4.105560302734375] # a proper initial gesture
            self.agent.set_joint_positions(self.initial_joint_positions)
        self.pr.step()
        self.initial_tip_positions = self.agent_ee_tip.get_position()
        self.initial_target_positions = self.target.get_position()

    def _get_state(self):
        '''
        Return state containing arm joint positions/velocities & target position.
        '''
        return np.array(self.agent.get_joint_positions() +  # list, dim=7
                self.agent.get_joint_velocities() +  # list, dim=7
                self.target.get_position())  # list, dim=3

    def _is_holding(self):
        '''
         Return the state of holding the target or not, return bool.
        '''
        # Note that the collision check is not always accurate all the time, 
        # for continuous collision frames, maybe only the first 4-5 frames of collision can be detected.
        pad_collide_object = self.gripper_left_pad.check_collision(self.target)
        if  pad_collide_object and self.proximity_sensor.is_detected(self.target)==True:
            return True 
        else:
            return False


    def _move(self, action, bounding_offset=0.15, step_factor=0.2, max_itr=20, max_error=0.05, rotation_norm =5.):
        ''' 
        Move the end effector on robot arm according to the action with inverse kinematics for 'end_position' control mode;
        Inverse kinematics mode control is achieved through setting the tip target instead of using .solve_ik(), 
        because sometimes the .solve_ik() does not function correctly.
        Mode: a close-loop proportional control, using ik.

        parameters:
        :bounding_offset: offset of bounding box outside the valid target position range, as valid and safe range of action
        :step_factor: small step factor mulitplied on the difference of current and desired position, i.e. proportional factor
        :max_itr: maximum moving iterations
        :max_error: upper bound of distance error for movement at each call
        :rotation_norm: factor for normalization of rotation values, since the action are of the same scale for each dimension
        '''
        pos=self.gripper.get_position()  

        # check if state+action will be within of the bounding box, if so, move normally; otherwise the action is not conducted.
        #  i.e. x_min < x < x_max  and  y_min < y < y_max  and  z > z_min
        if pos[0]+action[0]>POS_MIN[0]-bounding_offset and pos[0]+action[0]<POS_MAX[0]+bounding_offset  \
            and pos[1]+action[1] > POS_MIN[1]-bounding_offset and pos[1]+action[1] < POS_MAX[1]+2*bounding_offset  \
            and pos[2]+action[2] > POS_MIN[2]-2*bounding_offset:  # larger offset in z axis

            # there is a mismatch between the object set_orientation() and get_orientation():
            # the (x,y,z) in set_orientation() will be (y,x,-z) in get_orientation().
            ori_z=-self.agent_ee_tip.get_orientation()[2] # the minus is because the mismatch between the set_orientation() and get_orientation()
            target_pos = np.array(self.agent_ee_tip.get_position())+np.array(action[:3])
            diff=1  # intialization
            itr=0
            while np.sum(np.abs(diff))>max_error and itr<max_itr:
                itr+=1
                # set pos in small step
                cur_pos = self.agent_ee_tip.get_position()
                diff=target_pos-cur_pos  # difference of current and target position, close-loop control
                pos = cur_pos+step_factor*diff   # step small step according to current difference, to prevent that ik cannot be solved
                self.tip_target.set_position(pos.tolist())
                self.pr.step()  # every time when setting target tip, need to call simulation step to achieve it

            # one big step for z-rotation is enough, but small error still exists due to the ik solver
            ori_z+=rotation_norm*action[3]  # normalize the rotation values, as usually same action range is used in policy for both rotation and position
            self.tip_target.set_orientation([0, np.pi, ori_z])  # make gripper face downwards and rotate ori_z along z axis
            self.pr.step()

        else:
            print("Potential Movement Out of the Bounding Box!")
            pass # no action if potentially moving out of the bounding box

    def reinit(self):
        '''
        Reinitialize the environment, e.g. when the gripper is broken during exploration.
        '''
        self.shutdown()  # shutdown the original env first
        self.__init__(self.headless)  # initialize with the same headless mode


    def reset(self, random_target=False):
        '''
        Get a random position within a cuboid and set the target position.
        '''
        # set target object
        if random_target:  # randomize
            pos = list(np.random.uniform(POS_MIN, POS_MAX))  # sample from uniform in valid range
            self.target.set_position(pos)  # random position
        else:  # non-randomize
            self.target.set_position(self.initial_target_positions) # fixed position
        self.target.set_orientation([0,0,0])
        self.pr.step()

        # set end position to be initialized
        if self.control_mode == 'end_position':  # JointMode.IK
            self.agent.set_control_loop_enabled(True)  # ik mode
            self.tip_target.set_position(self.initial_tip_positions)  # cannot set joint positions directly due to in ik mode or force/torch mode
            self.pr.step()
            # prevent stuck cases. as using ik for moving, stucking can make ik cannot be solved therefore not reset correctly, therefore taking
            # some random action when desired position is not reached.
            itr=0
            max_itr=10
            while np.sum(np.abs(np.array(self.agent_ee_tip.get_position()-np.array(self.initial_tip_positions))))>0.1 and itr<max_itr:
                itr+=1
                self.step(np.random.uniform(-0.2,0.2,4))  # take random actions for preventing the stuck cases
                self.pr.step()

        elif self.control_mode == 'joint_velocity': # JointMode.FORCE
            self.agent.set_joint_positions(self.initial_joint_positions) 
            self.pr.step()

        # set collidable, for collision detection
        self.gripper_left_pad.set_collidable(True)  # set the pad on the gripper to be collidable, so as to check collision
        self.target.set_collidable(True)
        # open the gripper if it's not fully open
        if np.sum(self.gripper.get_open_amount())<1.5:
            self.gripper.actuate(1, velocity=0.5)  
            self.pr.step()

        return self._get_state()  # return current state of the environment

    def step(self, action):
        '''
        Move the robot arm according to the action.
        If control_mode=='joint_velocity', action is 7 dim of joint velocity values + 1 dim rotation of gripper;
        if control_mode=='end_position', action is 3 dim of tip (end of robot arm) position values + 1 dim rotation of gripper;
        '''
        # initialization
        done=False  # episode finishes
        reward=0
        hold_flag=False  # holding the object or not
        if self.control_mode == 'end_position':
            if action is None or action.shape[0]!=4:  # check if action is valid
                print('No actions or wrong action dimensions!')
                action = list(np.random.uniform(-0.1, 0.1, 4))  # random
            self._move(action)

        elif self.control_mode == 'joint_velocity':
            if action is None or action.shape[0]!=7:  # check if action is valid
                print('No actions or wrong action dimensions!')
                action = list(np.random.uniform(-0.1, 0.1, 7))  # random
            self.agent.set_joint_target_velocities(action)  # Execute action on arm
            self.pr.step()
      
        else:
            raise NotImplementedError

        ax, ay, az = self.gripper.get_position()
        if math.isnan(ax):  # capture the broken gripper cases during exploration
            print('Gripper position is nan.')
            self.reinit()
            done=True
        tx, ty, tz = self.target.get_position()
        offset=0.08   # augmented reward: offset of target position above the target object
        sqr_distance = (ax - tx) ** 2 + (ay - ty) ** 2 + (az - (tz+offset)) ** 2  # squared distance between the gripper and the target object
        
        ''' for visual-based control only, large time consumption! '''
        # current_vision = self.vision_sensor.capture_rgb()  # capture a screenshot of the view with vision sensor
        # plt.imshow(current_vision)
        # plt.savefig('./img/vision.png')
        

        # close the gripper if close enough to the object and the object is detected with the proximity sensor
        if sqr_distance<0.1 and self.proximity_sensor.is_detected(self.target)== True: 
            # make sure the gripper is open before grasping
            self.gripper.actuate(1, velocity=0.5)
            self.pr.step()
            self.gripper.actuate(0, velocity=0.5)  # if done, close the hand, 0 for close and 1 for open; velocity 0.5 ensures the gripper to close with in one frame
            self.pr.step()  # Step the physics simulation

            if self._is_holding():
                reward += self.reward_offset  # extra reward for grasping the object
                done=True
                hold_flag = True
            else:
                self.gripper.actuate(1, velocity=0.5)
                self.pr.step()
        elif np.sum(self.gripper.get_open_amount())<1.5: # if gripper is closed (not fully open) due to collision or esle, open it; get_open_amount() return list of gripper joint values
            self.gripper.actuate(1, velocity=0.5)
            self.pr.step()
        else:
            pass

        # the base reward is negative distance to target
        reward -= np.sqrt(sqr_distance) 
        
        # case when the object fall off the table
        if tz < self.initial_target_positions[2]-self.fall_down_offset:  
            done = True
            reward = -self.reward_offset

        # Augmented reward for orientation: better grasping gesture if the gripper has vertical orientation to the target object.
        # Note: the frame of gripper has a difference of pi/2 in z orientation as the frame of target.
        desired_orientation = np.concatenate(([np.pi, 0], [self.target.get_orientation()[2]]))  # gripper vertical to target in z and facing downwards, 
        rotation_penalty = -np.sum(np.abs(np.array(self.agent_ee_tip.get_orientation())-desired_orientation)) 
        rotation_norm = 0.02
        reward += rotation_norm*rotation_penalty

        # Penalty for collision with the table
        if self.gripper_left_pad.check_collision(self.table):
            reward -= self.penalty_offset
            #print('Penalize collision with table.')

        if math.isnan(reward):  # capture the cases of numerical problem
            reward = 0.

        return self._get_state(), reward, done, {'finished': hold_flag}

    def shutdown(self):
        ''' Close the simulator '''
        self.pr.stop()
        self.pr.shutdown()

if __name__ == '__main__':
    CONTROL_MODE='joint_velocity'  # 'end_position' or 'joint_velocity'
    env=GraspEnv(headless=False, control_mode=CONTROL_MODE)
    for eps in range(30):
        env.reset()
        for step in range(30):
            if CONTROL_MODE=='end_position':
                action=np.random.uniform(-0.2,0.2,4)  #  4 dim control for 'end_position': 3 positions and 1 rotation (z-axis)
            elif CONTROL_MODE=='joint_velocity':
                action=np.random.uniform(-2.,2.,7)
            else:
                raise NotImplementedError
            try:
                env.step(action)
            except KeyboardInterrupt:
                print('Shut Down!')
                env.shutdown()

    env.shutdown()
