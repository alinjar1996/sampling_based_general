import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory

import os
import time
import json
import csv
import numpy as np

import mujoco
from mujoco import viewer

from sampling_based_planner.mpc_planner import run_cem_planner
from sampling_based_planner.quat_math import *

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

PACKAGE_DIR = get_package_share_directory('real_demo')
np.set_printoptions(precision=4, suppress=True)

# target_positions = np.array([
#     [-0.2, 0.0, 0.3],
#     [-0.22, 0.1, 0.25],
#     [-0.25, -0.2, 0.3],
#     [-0.25, -0.25, 0.3]
# ])

# init_positions = np.array([
#     [-0.22, 0.0, 0.2],
#     [-0.22, 0.1, 0.2],
#     [-0.25, -0.2, 0.2],
#     [-0.28, -0.25, 0.2]
# ])


class Planner(Node):
    def __init__(self):
        super().__init__('planner')

        # Declare all parameters
        self.declare_parameter('use_hardware', False)
        self.declare_parameter('record_data', False)
        self.declare_parameter('idx', 0)
        self.declare_parameter('num_batch', 500)
        self.declare_parameter('num_steps', 15)
        self.declare_parameter('maxiter_cem', 1)
        self.declare_parameter('maxiter_projection', 5)
        self.declare_parameter('w_pos', 3.0)
        self.declare_parameter('w_rot', 0.5)
        self.declare_parameter('w_col', 500.0)
        self.declare_parameter('num_elite', 0.05)
        self.declare_parameter('timestep', 0.1)
        self.declare_parameter('position_threshold', 0.06)
        self.declare_parameter('rotation_threshold', 0.1)

        # Demo params
        self.use_hardware = self.get_parameter('use_hardware').get_parameter_value().bool_value
        self.record_data_ = self.get_parameter('record_data').get_parameter_value().bool_value
        self.idx = self.get_parameter('idx').get_parameter_value().integer_value
        self.idx = str(self.idx).zfill(2)

        # Planner params
        self.num_dof = 6
        self.init_joint_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        num_batch = self.get_parameter('num_batch').get_parameter_value().integer_value
        num_steps = self.get_parameter('num_steps').get_parameter_value().integer_value
        maxiter_cem = self.get_parameter('maxiter_cem').get_parameter_value().integer_value
        maxiter_projection = self.get_parameter('maxiter_projection').get_parameter_value().integer_value
        w_pos = self.get_parameter('w_pos').get_parameter_value().double_value
        w_rot = self.get_parameter('w_rot').get_parameter_value().double_value
        w_col = self.get_parameter('w_col').get_parameter_value().double_value
        num_elite = self.get_parameter('num_elite').get_parameter_value().double_value
        self.timestep = self.get_parameter('timestep').get_parameter_value().double_value
        position_threshold = self.get_parameter('position_threshold').get_parameter_value().double_value
        rotation_threshold = self.get_parameter('rotation_threshold').get_parameter_value().double_value
        self.num_targets = 21

        if self.record_data_:
            self.pathes = {
                "setup": os.path.join(PACKAGE_DIR, 'data', 'planner', 'setup', f'setup_{self.idx}.npz'),
                "trajectory": os.path.join(PACKAGE_DIR, 'data', 'planner', 'trajectory', f'traj_{self.idx}.npz'),
                # "benchmark": os.path.join(PACKAGE_DIR, 'data', 'planner', 'benchmark', f'bench_{num_batch}_{num_steps}_walker{self.idx}.npz'),
            }
            self.data_buffers = {
                'batch_size': [num_batch],
                'horizon': [num_steps],

                # 'target_0': [0]*self.num_targets,
                # 'total_time_s': [0]*self.num_targets,
                # 'success': [0]*self.num_targets,
                # 'reason': [0]*self.num_targets,

                # 'step_time_ms': [[] for _ in range(self.num_targets)],
                # 'theta': [[] for _ in range(self.num_targets)],
                # 'thetadot': [[] for _ in range(self.num_targets)],

                # 'cost_r': [[] for _ in range(self.num_targets)],
                # 'cost_eef_to_obj': [[] for _ in range(self.num_targets)],
                # 'cost_obj_to_targ': [[] for _ in range(self.num_targets)],
                # 'cost_dist': [[] for _ in range(self.num_targets)],
                # 'cost_zy': [[] for _ in range(self.num_targets)],
            }



        self.target_idx = 0

        cost_weights = {
            'height': 100.0,
			'orientation': 10.0,
            'velocity': 100.0,
            'control': 0.0
        }


        self.thetadot = np.zeros(self.num_dof)

        # Initialize robot connection
        self.rtde_c_0 = None
        self.rtde_r_0 = None

        self.rtde_c_1 = None
        self.rtde_r_1 = None

        self.grippers = {
            '0': {
                'srv': None,
                'state': 'open'
            },
            '1': {
                'srv': None,
                'state': 'open'
            }
        }

        if self.use_hardware:
            self.initialize_robot_connection()
        
        # Initialize MuJoCo model and data
        model_path = os.path.join(get_package_share_directory('real_demo'), 'walker_mjx', 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.timestep
  
        joint_names_pos = list()
        joint_names_vel = list()
        for i in range(self.model.njnt):
            joint_type = self.model.jnt_type[i]
            n_pos = 7 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 4 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            n_vel = 6 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            
            for _ in range(n_pos):
                joint_names_pos.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
            for _ in range(n_vel):
                joint_names_vel.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
        
        
        # robot_joints = np.array(['shoulder_pan_joint_1', 'shoulder_lift_joint_1', 'elbow_joint_1', 'wrist_1_joint_1', 'wrist_2_joint_1', 'wrist_3_joint_1',
        #                         'shoulder_pan_joint_2', 'shoulder_lift_joint_2', 'elbow_joint_2', 'wrist_1_joint_2', 'wrist_2_joint_2', 'wrist_3_joint_2'])
        
        robot_joints = np.array(['right_hip', 'right_knee', 
                        'right_ankle', 'left_hip', 
                        'left_knee', 'left_ankle'])
        self.joint_mask_pos = np.isin(joint_names_pos, robot_joints)
        self.joint_mask_vel = np.isin(joint_names_vel, robot_joints)

        # print("self.joint_mask_vel", self.joint_mask_vel)


        if self.use_hardware:
            setup = np.load(os.path.join(PACKAGE_DIR, 'data', 'manual', 'setup', f'setup_000.npz'), allow_pickle=True)

            marker_pos = setup['setup'][0][1]
            marker_diff = marker_pos-self.model.body(name='table0_marker').pos


        # mujoco.mj_forward(self.model, self.data)
        self.data = mujoco.MjData(self.model)

        
        mujoco.mj_forward(self.model, self.data)

        self.data.qpos[self.joint_mask_pos] = self.init_joint_position


        self.traj_time_start = time.time()
        self.success = 0
        self.reason = 'na'
        

        # Initialize CEM/MPC planner
        self.planner = run_cem_planner(
            model=self.model,
            data=self.data,
            num_dof=self.num_dof,
            num_batch=num_batch,
            num_steps=num_steps,
            maxiter_cem=maxiter_cem,
            maxiter_projection=maxiter_projection,
            num_elite=num_elite,
            timestep=self.timestep,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            cost_weights=cost_weights
        )
        
        # Setup viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.cam.distance = 4.0 
        self.viewer.cam.azimuth = 90.0 
        self.viewer.cam.elevation = -30.0 

        # Setup subscribers
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.subscription_object0 = self.create_subscription(PoseStamped, '/vrpn_mocap/object1/pose', self.object0_callback, qos_profile)
        self.subscription_obstacle0 = self.create_subscription(PoseStamped, '/vrpn_mocap/obstacle1/pose', self.obstacle0_callback, qos_profile)
        
        # Start control timer
        self.timer = self.create_timer(self.timestep, self.control_loop)


    def render_trace(self, viewer_, torso_trace_positions):
        """Render the end-effector trajectory trace in the viewer."""
        # Clear any existing overlay geoms
        viewer_.user_scn.ngeom = 0

        # for pos in torso_trace_positions:   # each pos is already [x,y,z]
        #     # Create a new geom in the user scene
        #     geom_id = viewer_.user_scn.ngeom
        #     viewer_.user_scn.ngeom += 1

        #     # Ensure correct numpy types for MuJoCo
        #     size = np.array([0.02, 0.02, 0.02], dtype=np.float64)
        #     pos = np.array(pos, dtype=np.float64).reshape(3)
        #     mat = np.eye(3, dtype=np.float64).flatten()
        #     rgba = np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float32)

        #     # Initialize the geom properties
        #     mujoco.mjv_initGeom(
        #         viewer_.user_scn.geoms[geom_id],
        #         mujoco.mjtGeom.mjGEOM_SPHERE,
        #         size,
        #         pos,
        #         mat,
        #         rgba
        #     )

        for i, pos in enumerate(torso_trace_positions):
            geom_id = viewer_.user_scn.ngeom
            viewer_.user_scn.ngeom += 1

            size = np.array([0.02, 0.02, 0.02], dtype=np.float64)
            pos = np.array(pos, dtype=np.float64).reshape(3)
            mat = np.eye(3, dtype=np.float64).flatten()

            # Interpolation factor [0..1]
            t = i / (len(torso_trace_positions) - 1 + 1e-9)

            # Fade from solid red → transparent red
            start_color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)   # opaque red
            end_color   = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)   # transparent red
            rgba = (1 - t) * start_color + t * end_color

            mujoco.mjv_initGeom(
                viewer_.user_scn.geoms[geom_id],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                pos,
                mat,
                rgba
            )


    def control_loop(self):
        """Main control loop running at fixed interval"""
        start_time = time.time()

           
    
        # Get current state
        if self.use_hardware:
            current_pos_0 = np.array(self.rtde_r_0.getActualQ())
            current_pos_1 = np.array(self.rtde_r_1.getActualQ())

            current_pos = np.concatenate((current_pos_0, current_pos_1), axis=None)
            # current_pos = self.data.qpos[self.joint_mask_pos]
            current_vel = self.thetadot
        else:
            current_pos = self.data.qpos[self.joint_mask_pos]
            current_vel = self.thetadot
        
        
        # Compute control
        self.thetadot, cost, cost_list, thetadot_horizon, theta_horizon, torso_trace_planned = self.planner.compute_control(current_pos, current_vel)
        cost_height, cost_orientation, cost_velocity, cost_control = cost_list
        
        print("self.thetadot", self.thetadot)

        if self.use_hardware:
            # Send velocity command
            self.rtde_c_0.speedJ(self.thetadot[:self.planner.num_dof//2], acceleration=1, time=0.1)
            self.rtde_c_1.speedJ(self.thetadot[self.planner.num_dof//2:], acceleration=1, time=0.1)

            # Update MuJoCo state
            current_pos = np.concatenate((np.array(self.rtde_r_0.getActualQ()), np.array(self.rtde_r_1.getActualQ())), axis=None)
            self.data.qpos[self.joint_mask_pos] = current_pos
            self.data.qvel[:] = np.zeros(len(self.joint_mask_vel))
            self.data.qvel[self.joint_mask_vel] = self.thetadot
            mujoco.mj_step(self.model, self.data)
        else:
            self.data.qvel[:] = np.zeros(len(self.joint_mask_vel))
            self.data.qvel[self.joint_mask_vel] = self.thetadot
            mujoco.mj_step(self.model, self.data)
        
        # print("torso_trace_planned", torso_trace_planned)

        self.render_trace(self.viewer, torso_trace_planned[:,:3])


        # Update viewer
        self.viewer.sync()
        
        # Print debug info
        print(f'\n| Target idx: {self.target_idx} '
              f'\n| Total time: {"%.0f"%(time.time() - self.traj_time_start)}s '
              f'\n| Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms '
              f'\n| Cost: {np.round(cost, 2)} '
              f'\n| Cost_Height: {np.round(cost_height, 2)} ',
              f'\n| Cost_Orientation: {np.round(cost_orientation, 2)} ', 
              f'\n| Cost_Velocity: {np.round(cost_velocity, 2)} ', 
              f'\n| Cost_Control: {np.round(cost_control, 2)} ', flush=True)
        
        time_until_next_step = self.model.opt.timestep - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step) 

    
    def close_connection(self):
        if self.use_hardware:
            """Cleanup robot connection"""
            if self.rtde_c_0:
                self.rtde_c_0.speedStop()
                self.rtde_c_0.disconnect()

            if self.rtde_c_1:
                self.rtde_c_1.speedStop()
                self.rtde_c_1.disconnect()
            print("Disconnected from UR5e Robot", flush=True)

    def object0_callback(self, msg):
        """Callback for target object pose updates"""

        pose = msg.pose
        ball_pos = np.array([-pose.position.x, -pose.position.y, pose.position.z-0.05])
        self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='ball').id]] = ball_pos
            # mujoco.mj_forward(self.model, self.data)
            # self.planner.update_targets(target_idx=0, target_pos=self.data.xpos[self.model.body(name="target_00").id], target_rot=self.data.xquat[self.model.body(name="target_00").id])
            # self.planner.update_targets(target_idx=1, target_pos=self.data.xpos[self.model.body(name="target_11").id], target_rot=self.data.xquat[self.model.body(name="target_11").id])
            # self.planner.target_2[:3] = ball_pos

    def obstacle0_callback(self, msg):
        """Callback for obstacle pose updates"""
        pose = msg.pose
        obstacle_pos = np.array([-pose.position.x, -pose.position.y, pose.position.z])
        obstacle_rot = np.array([0.0, 1.0, 0, 0])
        self.planner.update_obstacle(obstacle_pos, obstacle_rot)

    # def record_data(self):
    #     """Save data to npy file"""
        

    #     np.savez(
    #         self.pathes['benchmark'],
    #         batch_size=np.array(self.data_buffers['batch_size']),
    #         horizon=np.array(self.data_buffers['horizon']),
    #         total_time=np.array(self.data_buffers['total_time_s']),
    #         step_time=np.array(self.data_buffers['step_time_ms'], dtype=object),
    #         success=np.array(self.data_buffers['success']),
    #         reason=np.array(self.data_buffers['reason']),
    #         target_0=np.array(self.data_buffers['target_0']),
    #         theta=np.array(self.data_buffers['theta'], dtype=object),
    #         thetadot=np.array(self.data_buffers['thetadot'], dtype=object),
    #         cost_r=np.array(self.data_buffers['cost_r'], dtype=object),
    #         cost_eef_to_obj=np.array(self.data_buffers['cost_eef_to_obj'], dtype=object),
    #         cost_obj_to_targ=np.array(self.data_buffers['cost_obj_to_targ'], dtype=object),
    #         cost_dist=np.array(self.data_buffers['cost_dist'], dtype=object),
    #         cost_zy=np.array(self.data_buffers['cost_zy'], dtype=object),
    #     )
    #     self.data_saved = True
    #     print("Saving data...")

def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    print("Initialized node.", flush=True)
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        print("Shutting down...", flush=True)
    finally:
        # if rclpy.ok():
        if planner.record_data_:
            planner.record_data()
        planner.close_connection()
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()