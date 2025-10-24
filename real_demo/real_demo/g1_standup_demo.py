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
        self.declare_parameter('num_elite', 0.05)
        self.declare_parameter('timestep', 0.1)

        # Demo params
        self.use_hardware = self.get_parameter('use_hardware').get_parameter_value().bool_value
        self.record_data_ = self.get_parameter('record_data').get_parameter_value().bool_value
        self.idx = self.get_parameter('idx').get_parameter_value().integer_value
        self.idx = str(self.idx).zfill(2)

        # Planner params
        self.num_dof = 29
        self.init_joint_position = np.array([0.0]*self.num_dof)
        num_batch = self.get_parameter('num_batch').get_parameter_value().integer_value
        num_steps = self.get_parameter('num_steps').get_parameter_value().integer_value
        maxiter_cem = self.get_parameter('maxiter_cem').get_parameter_value().integer_value
        maxiter_projection = self.get_parameter('maxiter_projection').get_parameter_value().integer_value
        num_elite = self.get_parameter('num_elite').get_parameter_value().double_value
        self.timestep = self.get_parameter('timestep').get_parameter_value().double_value


        

        self.pathes = {
        "setup": os.path.join(
            PACKAGE_DIR, "data", "planner", "setup",
            f"setup_g1_standup_{num_batch}_{num_steps}_{maxiter_cem}_"
            f"{maxiter_projection}_{int(self.timestep*1000)}_{int(num_elite*100)}_{self.idx}.npz"
        ),
        "trajectory": os.path.join(
            PACKAGE_DIR, "data", "planner", "trajectory",
            f"traj_g1_standup_{num_batch}_{num_steps}_{maxiter_cem}_"
            f"{maxiter_projection}_{int(self.timestep*1000)}_{int(num_elite*100)}_{self.idx}.npz"
        )}

        self.data_buffers = {
                'batch_size': [num_batch],
                'horizon': [num_steps],
                'theta': [],           # Joint positions over time
                'control': [],          # control commands over time  
                'cost_cem': [],        # CEM cost over time
                'cost_height_cem': [],      # Theta cost component
                'cost_orientation_cem': [],      # Orientation cost component
                'cost_nominal_cem': [],    # Control cost component
                'total_time': [],       # Timestamps
                'theta_horizon': [],   # Planned theta horizon 
                'control_horizon': [], # Planned control horizon
                'control_samples': [], # control sample
                'control_filtered': [], # control filtered
                'torso_trace_planned': [], # best torso_trace
                'torso_trace_all': [], # torso_trace_all_samples
                'primal_res': [], # Primal residual
                'fixed_res': [], # Fixed point residual
                'xi_samples': [], # xi_samples
            }
            

        cost_weights = {
            'orientation': 20.0,
			'height': 15.0,
            'nominal': 15.0
        }


        self.control = np.zeros(self.num_dof)
        self.control_array = np.zeros((num_steps, self.num_dof))

        
        # Initialize MuJoCo model and data
        model_path = os.path.join(get_package_share_directory('real_demo'), 'g1_mjx', 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.timestep

         # Get sensor ids (similar to your cem_planner example)
                # Get sensor and site ids
        self.orientation_sensor_id = self.model.sensor("imu_in_torso_quat").id
        self.velocity_sensor_id = self.model.sensor("imu_in_torso_linvel").id
        self.torso_id = self.model.site("imu_in_torso").id

        
        print(f"Sensor IDs - Orientation: {self.orientation_sensor_id}, "
              f"Velocity: {self.velocity_sensor_id}, "
              f"Torso: {self.torso_id}")
        
        robot_joints_left_leg = np.array(['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
                                          'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint'])
        robot_joints_right_leg = np.array(['right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
                                           'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'])
        robot_joints_waist = np.array(['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'])
        robot_joints_left_arm = np.array(['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                                          'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint'])
        
        robot_joints_right_arm = np.array(['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                                          'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'])

        self.robot_joints = np.concatenate([
            robot_joints_left_leg,
            robot_joints_right_leg,
            robot_joints_waist,
            robot_joints_left_arm,
            robot_joints_right_arm
        ])

        # self.robot_joints = self.planner.cem.robot_joints
        
        joint_names_pos = list()
        joint_names_vel = list()
        joint_names_ctrl = list()
        for i in range(self.model.njnt):
            joint_type = self.model.jnt_type[i]
            n_pos = 7 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 4 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            n_vel = 6 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            n_ctrl = 6 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            
            for _ in range(n_pos):
                joint_names_pos.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
            for _ in range(n_vel):
                joint_names_vel.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
            for _ in range(n_ctrl):
                joint_names_ctrl.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
        
        
        # robot_joints = np.array(['shoulder_pan_joint_1', 'shoulder_lift_joint_1', 'elbow_joint_1', 'wrist_1_joint_1', 'wrist_2_joint_1', 'wrist_3_joint_1',
        #                         'shoulder_pan_joint_2', 'shoulder_lift_joint_2', 'elbow_joint_2', 'wrist_1_joint_2', 'wrist_2_joint_2', 'wrist_3_joint_2'])
        

                         
        self.joint_mask_pos = np.isin(joint_names_pos, self.robot_joints)
        self.joint_mask_vel = np.isin(joint_names_vel,self. robot_joints)
        self.joint_mask_ctrl = np.isin(joint_names_ctrl, self.robot_joints)
        self.joint_ctrl_indices = jnp.where(self.joint_mask_ctrl)[0]
        self.actuator_joint_ids = self.model.actuator_trnid[:, 0]
        self.actuator_ctrl_indices = [
			i for i, j in enumerate(self.actuator_joint_ids)
			if self.joint_mask_ctrl[j]
		]

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
            cost_weights=cost_weights
        )
        
        # Setup viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.cam.distance = 5.0 
        self.viewer.cam.azimuth = 90.0 
        self.viewer.cam.elevation = -30.0 

        # Setup subscribers
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        # self.subscription_object0 = self.create_subscription(PoseStamped, '/vrpn_mocap/object1/pose', self.object0_callback, qos_profile)
        # self.subscription_obstacle0 = self.create_subscription(PoseStamped, '/vrpn_mocap/obstacle1/pose', self.obstacle0_callback, qos_profile)
        
        # Start control timer
        self.timer = self.create_timer(self.timestep, self.control_loop)


    def render_trace(self, viewer_, torso_trace_positions):
        """Render the end-effector trajectory trace in the viewer."""
        # Clear any existing overlay geoms
        viewer_.user_scn.ngeom = 0

        for i, pos in enumerate(torso_trace_positions):
            geom_id = viewer_.user_scn.ngeom
            viewer_.user_scn.ngeom += 1

            size = np.array([0.02, 0.02, 0.02], dtype=np.float64)
            pos = np.array(pos, dtype=np.float64).reshape(3)
            mat = np.eye(3, dtype=np.float64).flatten()

            # Interpolation factor [0..1]
            t = i / (len(torso_trace_positions) - 1 + 1e-9)

            # Fade from solid red → transparent red
            start_color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)   #  red
            end_color   = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)   # green
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
        
        current_pos = self.data.qpos[self.joint_mask_pos]
        current_vel = self.data.qvel[self.joint_mask_vel]
        # self.control = np.mean(self.control_array[1:int(0.5*self.planner.num_steps)], axis = 0)
        self.control = np.mean(self.control_array[1:5], axis = 0)
        current_control = self.control

        print("self.control", self.control)
        
        
        # Compute control
        (self.control_array, 
         cost_cem, 
         cost_list_cem, 
         control_horizon, 
         theta_horizon, 
         torso_trace_planned,
         torso_trace_all,
         control_samples,
         control_filtered,
         primal_res,
         fixed_res,
         xi_samples) = self.planner.compute_control(self.data, current_pos, current_vel, current_control)
        
        cost_orientation_cem, cost_height_cem, cost_nominal_cem = (
                    cost_list_cem[:, 0], 
                    cost_list_cem[:, 1], 
                    cost_list_cem[:, 2]
                )
        
        
        cost_orientation, cost_height, cost_nominal = cost_list_cem[-1]
        
        # # Get the torso position from the sensor data
        # torso_pos = self.get_sensor_value(self.torso_position_sensor)

        # if self.viewer:
        #     self.viewer.cam.lookat[:] = torso_pos
        
        

        # STORE THE DATA
        if self.record_data_:
            current_time = time.time() - self.traj_time_start
            
            self.data_buffers['theta'].append(current_pos.copy())
            # self.data_buffers['control'].append(self.control.copy())
            self.data_buffers['control'].append(np.atleast_1d(np.squeeze(self.control.copy())))
            self.data_buffers['cost_cem'].append(cost_cem.copy())
            self.data_buffers['cost_height_cem'].append(cost_height_cem)
            self.data_buffers['cost_orientation_cem'].append(cost_orientation_cem)
            self.data_buffers['cost_nominal_cem'].append(cost_nominal_cem)
            self.data_buffers['total_time'].append(current_time)
            self.data_buffers['theta_horizon'].append(theta_horizon.copy())
            self.data_buffers['control_horizon'].append(control_horizon.copy())
            self.data_buffers['control_samples'].append(control_samples.copy())
            self.data_buffers['control_filtered'].append(control_filtered.copy())
            self.data_buffers['torso_trace_planned'].append(torso_trace_planned.copy())
            self.data_buffers['torso_trace_all'].append(torso_trace_all.copy())
            self.data_buffers['primal_res'].append(primal_res.copy())
            self.data_buffers['fixed_res'].append(fixed_res.copy())
            self.data_buffers['xi_samples'].append(xi_samples.copy())

        
        
        # self.data.ctrl[self.actuator_ctrl_indices] = self.control
        self.data.ctrl[self.planner.cem.actuator_ctrl_indices] = self.control
        

        mujoco.mj_step(self.model, self.data)
        

        # Print sensor data
        self.print_sensor_data()

        self.render_trace(self.viewer, torso_trace_planned[:,:3])


        # Update viewer
        self.viewer.sync()
        
        # Print debug info
        print(f'\n| Total time: {"%.0f"%(time.time() - self.traj_time_start)}s '
              f'\n| Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms '
              f'\n| Cost: {np.round(cost_cem[-1], 2)} '
              f'\n| Cost_Height: {np.round(cost_height, 2)} ',
              f'\n| Cost_Orientation: {np.round(cost_orientation, 2)} ', 
              f'\n| cost_nominal: {np.round(cost_nominal, 2)} ', flush=True)
        
        print("=" * 40)
        time_until_next_step = self.model.opt.timestep - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step) 

    def save_data(self):
        """Save all recorded data to file"""
        if not self.record_data_ or not self.data_buffers['theta']:
            print("No data to save or recording disabled")
            return
            
        # Convert lists to numpy arrays for efficient storage
        save_dict = {
            'batch_size': np.array(self.data_buffers['batch_size']),
            'horizon': np.array(self.data_buffers['horizon']),
            'theta': np.array(self.data_buffers['theta']),
            'control': np.array(self.data_buffers['control']),
            'cost_cem': np.array(self.data_buffers['cost_cem']),
            'cost_height_cem': np.array(self.data_buffers['cost_height_cem']),
            'cost_orientation_cem': np.array(self.data_buffers['cost_orientation_cem']),
            'cost_nominal_cem': np.array(self.data_buffers['cost_nominal_cem']),
            'total_time': np.array(self.data_buffers['total_time']),
            'theta_horizon': np.array(self.data_buffers['theta_horizon']),
            'control_horizon': np.array(self.data_buffers['control_horizon']),
            'control_samples': np.array(self.data_buffers['control_samples']),
            'control_filtered': np.array(self.data_buffers['control_filtered']),
            'torso_trace_planned': np.array(self.data_buffers['torso_trace_planned']),
            'torso_trace_all': np.array(self.data_buffers['torso_trace_all']),
            'primal_res': np.array(self.data_buffers['primal_res']),
            'fixed_res': np.array(self.data_buffers['fixed_res']),
            'xi_samples': np.array(self.data_buffers['xi_samples']),
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.pathes["trajectory"]), exist_ok=True)
        
        # Save data
        np.savez(self.pathes["trajectory"], **save_dict)
        print(f"Data saved to {self.pathes['trajectory']}")
        print(f"Recorded {len(self.data_buffers['theta'])} time steps")
    
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
    

    # Extract sensor values using sensor_adr (same approach as in compute_cost_single)
    def get_sensor_value(self, sensor_id):
            
            sensor_data = self.data.sensordata
            sensor_adr = self.model.sensor_adr[sensor_id]
            sensor_dim = self.model.sensor_dim[sensor_id]
            if sensor_dim == 1:
                return sensor_data[sensor_adr]
            else:
                return sensor_data[sensor_adr:sensor_adr + sensor_dim]
            
    def print_sensor_data(self):
        """Print sensor data similar to your cem_planner example"""
        
        self.torso_id = self.model.site("imu_in_torso").id
        # Get sensor values
        torso_quat = self.get_sensor_value(self.orientation_sensor_id)
        torso_vel = self.get_sensor_value(self.velocity_sensor_id) 
        # torso_zaxis = self.get_sensor_value(self.torso_zaxis_sensor)
        
        print(f"\n=== MUJOCO SENSORS ===")
        print(f"Torso Quat: {torso_quat}")
        print(f"Torso Velocity: {torso_vel}")
        print("=" * 20)

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
            planner.save_data()
        planner.close_connection()
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()