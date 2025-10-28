import os
import sys
from ament_index_python.packages import get_package_share_directory


xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


from functools import partial
import numpy as np

import mujoco
import mujoco.mjx as mjx 
import jax
import jax.numpy as jnp
import xml.etree.ElementTree as ET

# Get the folder containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))  # if in a script
# current_dir = os.getcwd()  # if in Jupyter notebook

# Add parent folder to sys.path
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from math_utils.bernstein_coeff_ordern_arbitinterval import bernstein_coeff_ordern_new


class cem_planner():

	def __init__(self, model=None, num_dof=None, num_batch=None, num_steps=None, timestep=None, maxiter_cem=None, num_elite=None, 
			     maxiter_projection=None, max_joint_intcontrol = None ,max_joint_control = None, 
				 max_joint_dcontrol = None, max_joint_ddcontrol = None):
		super(cem_planner, self).__init__()
	    

		self.num_dof = num_dof
		self.num_batch = num_batch
		self.t = timestep
		self.num = num_steps
		self.num_elite = num_elite

		self.t_fin = self.num*self.t

		
		tot_time = np.linspace(0, self.t_fin, self.num)
		self.tot_time = tot_time
		tot_time_copy = tot_time.reshape(self.num, 1)

		# self.P = jnp.identity(self.num) # control mapping 
		# self.Pdot = jnp.diff(self.P, axis=0)/self.t # Dcontrol mapping
		# self.Pddot = jnp.diff(self.Pdot, axis=0)/self.t # DDcontrol mapping
		# self.Pint = jnp.cumsum(self.P, axis=0)*self.t # Intcontrol mapping
		
		self.P, self.Pdot, self.Pddot = bernstein_coeff_ordern_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

		self.Pint = jnp.zeros_like(self.P) 

		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
		self.Pint_jax = jnp.asarray(self.Pint)

		self.nvar_single = jnp.shape(self.P_jax)[1]
		self.nvar = self.nvar_single*self.num_dof 
  
		self.rho_ineq = 1.0
		self.rho_projection = 1.0

		self.A_projection = jnp.identity(self.nvar)

		A_pos_ineq, A_pos = self.get_A_pos()
		self.A_pos_ineq = jnp.asarray(A_pos_ineq) 
		self.A_pos = jnp.asarray(A_pos)

		A_vel_ineq, A_vel = self.get_A_vel()
		self.A_vel_ineq = jnp.asarray(A_vel_ineq) 
		self.A_vel = jnp.asarray(A_vel)

		A_acc_ineq, A_acc = self.get_A_acc()
		self.A_acc_ineq = jnp.asarray(A_acc_ineq)
		self.A_acc = jnp.asarray(A_acc)
  
		A_int_pos_ineq, A_int_pos = self.get_A_int_pos()
		self.A_int_pos_ineq = jnp.asarray(A_int_pos_ineq) 
		self.A_int_pos = jnp.asarray(A_int_pos)

		# Combined control matrix (like A_control in )
		self.A_control = jnp.vstack((
			self.A_pos_ineq,
			self.A_vel_ineq,
			self.A_acc_ineq
		#	self.A_int_pos_ineq
		))

		A_eq = self.get_A_eq()
		self.A_eq = jnp.asarray(A_eq)

		A_intpos, A_pos, A_vel, A_acc = self.get_A_control()

		self.A_intpos = np.asarray(A_intpos)
		self.A_pos = np.asarray(A_pos)
		self.A_vel = np.asarray(A_vel)
		self.A_acc = np.asarray(A_acc)
		
		self.key= jax.random.PRNGKey(42)
		self.maxiter_projection = maxiter_projection
		self.maxiter_cem = maxiter_cem

		# self.pos_max = max_joint_control
		self.vel_max = max_joint_dcontrol
		self.acc_max = max_joint_ddcontrol
			
		    
    	# Calculating number of Inequality constraints
		self.num_control   = self.P.shape[0]       # number of time samples for control 
		self.num_dcontrol  = self.Pdot.shape[0]    # number of samples for rate of change of control)
		self.num_ddcontrol = self.Pddot.shape[0]   # number of samples for double rate of change of control
		self.num_intcontrol = self.Pint.shape[0]   # number of samples for integrated control

		self.num_control_constraints = 2 * self.num_control * num_dof
		self.num_dcontrol_constraints = 2 * self.num_dcontrol * num_dof
		self.num_ddcontrol_constraints = 2 * self.num_ddcontrol * num_dof
		self.num_intcontrol_constraints = 2 * self.num_intcontrol * num_dof
		self.num_total_constraints = (self.num_control_constraints + self.num_dcontrol_constraints + 
								      self.num_ddcontrol_constraints)
		self.num_total_constraints_per_dof = self.num_total_constraints / self.num_dof

		self.ellite_num = int(self.num_elite*self.num_batch)



		self.alpha_mean = 0.6
		self.alpha_cov = 0.6

		self.lamda = 0.1
		self.g = 10
		self.vec_product = jax.jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

		self.gamma = 0.98 #Discount factor

		self.model = model
		self.data = mujoco.MjData(self.model)
		self.model.opt.timestep = self.t

		self.mjx_model = mjx.put_model(self.model)
		self.mjx_data = mjx.put_data(self.model, self.data)
		self.mjx_data = jax.jit(mjx.forward)(self.mjx_model, self.mjx_data)
		self.jit_step = jax.jit(mjx.step)
		self.jit_forward = jax.jit(mjx.forward)

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
		# 				'shoulder_pan_joint_2', 'shoulder_lift_joint_2', 'elbow_joint_2', 'wrist_1_joint_2', 'wrist_2_joint_2', 'wrist_3_joint_2'])
		
	
		# robot_joints = np.array(['right_hip', 'right_knee', 'right_ankle',
        #                  'left_hip', 'left_knee', 'left_ankle'])
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
		
		self.joint_mask_pos = np.isin(np.array(joint_names_pos), self.robot_joints)
		self.joint_mask_vel = np.isin(np.array(joint_names_vel), self.robot_joints)
		self.joint_mask_ctrl = np.isin(np.array(joint_names_ctrl), self.robot_joints)
		self.joint_ctrl_indices = jnp.where(self.joint_mask_ctrl)[0]

		# Find where your first controllable joint appears
		actuator_joint_ids_ = model.actuator_trnid[:, 0]
		offset_joint_id = np.where(self.joint_mask_ctrl)[0][0] - actuator_joint_ids_[0]
		actuator_joint_ids = model.actuator_trnid[:, 0] + offset_joint_id

		# actuator_joint_ids = self.model.actuator_trnid[:, 0]
		self.actuator_ctrl_indices = [
			i for i, j in enumerate(actuator_joint_ids)
			if self.joint_mask_ctrl[j]
		]

		# # Add this after your existing joint mask code in __init__
		# print("\n=== DEBUG: JOINT MASKS ===")
		# print("Position-controlled joints:")
		# for i, name in enumerate(joint_names_pos):
		# 	print(f"  {i}: '{name}' -> controlled: {self.joint_mask_pos[i]}")

		print("\nControlled joints:")  
		for i, name in enumerate(joint_names_ctrl):
			print(f"  {i}: '{name}' -> controlled: {self.joint_mask_ctrl[i]}")

		# print("\nRobot joints:", robot_joints)

		self.geom_ids = []
		
		for i in range(self.model.ngeom):
			name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
			if name is not None and (
				name.startswith('robot') 
				or
				name.startswith('object') 
				# name.startswith('target')
			):  
				print(f"Found geom: id={i}, name='{name}'")
				self.geom_ids.append(i)

		self.geom_ids_all = np.array(self.geom_ids)
		self.mask = jnp.any(jnp.isin(self.mjx_data.contact.geom, self.geom_ids_all), axis=1)

		self.mask_move = self.mask

		self.mask = jnp.tile(self.mask, (self.num, 1))
		
		# self.hande_id_0 = self.model.body(name="hande_0").id

		# Get sensor ids


		# Get sensor and site ids
		self.orientation_sensor_id = self.model.sensor("imu_in_torso_quat").id
		self.velocity_sensor_id = self.model.sensor("imu_in_torso_linvel").id
		self.torso_id = self.model.site("imu_in_torso").id

		# self.torso = self.model.site(name="torso_site").id

        # Set the target height
		self.target_height = 0.9

        # Standing configuration
		self.qstand = jnp.array(self.model.keyframe("stand").qpos)

		self.p_min, self.p_max = self.extract_joint_limits_from_model(self.model)

		# self.compute_rollout_batch = jax.vmap(self.compute_rollout_single, in_axes = (None, 0, None, None))
		self.compute_rollout_batch = jax.vmap(self.compute_rollout_single_control, in_axes = (None, 0, None, None))
		self.compute_cost_batch = jax.vmap(self.compute_cost_single, in_axes = (0, 0, 0, 0, None))
		self.compute_boundary_vec_batch = (jax.vmap(self.compute_boundary_vec_single, in_axes = (0)  )) # vmap parrallelization takes place over first axis
          
		self.print_info()


	def print_info(self):
		print(
			f'\n Default backend: {jax.default_backend()}'
			# f'\n Model path: {self.model_path}',
			f'\n Timestep: {self.t}',
			f'\n CEM Iter: {self.maxiter_cem}',
			f'\n Projection Iter: {self.maxiter_projection}',
			f'\n Number of batches: {self.num_batch}',
			f'\n Number of steps per trajectory: {self.num}',
			f'\n Time per trajectory: {self.t_fin}',
			f'\n Number of variables: {self.nvar}',
			f'\n Number of Total constraints: {self.num_total_constraints}',

		)
	
	def extract_joint_limits_from_model(self, model):
		"""
		Extracts u_min and u_max for all position-controlled joints directly 
		from the loaded MuJoCo model object, ensuring the order matches the actuators.

		Args:
			model: The loaded mujoco.MjModel instance (e.g., self.model).

		Returns:
			A tuple (u_min, u_max) of JAX arrays, shape (num_controlled_dof,).
		"""
		
		# 1. Get the ID of the kinematic element (joint) targeted by each actuator.
		# For position actuators, model.actuator_trnid[:, 0] gives the joint ID in actuator order (nu=29).
		# We slice up to model.nu to ensure we only consider defined actuators.
		joint_ids = model.actuator_trnid[:model.nu, 0].copy() 
		
		# 2. Check which of these joints are actually limited.
		# The jnt_range values are only meaningful if model.jnt_limited is True for that ID.
		is_limited = model.jnt_limited[joint_ids].astype(bool)
		
		# 3. Filter the IDs to only include the limited ones.
		# This ensures your final arrays only contain valid range data. (Should be all 29 joints for this robot).
		limited_joint_ids = joint_ids[is_limited]
		
		# 4. Extract limits from the full joint range table (model.jnt_range has shape (model.njnt, 2)).
		# The indices used here ensure the limits maintain the actuator order.
		u_min_np = model.jnt_range[limited_joint_ids, 0]
		u_max_np = model.jnt_range[limited_joint_ids, 1]
		
		# 5. Convert to JAX arrays
		u_min = jnp.array(u_min_np)
		u_max = jnp.array(u_max_np)
		
		return u_min, u_max

	# Example usage (assuming your XML is saved as 'robot_model.xml'):
	# self.u_min, self.u_max = extract_joint_limits_from_xml('robot_model.xml')
	def get_A_control(self):

		# This is valid while dealing with knots anfd projecting into pos,vel,acc space with Bernstein Polynomials
		# A_intpos = np.kron(np.identity(self.num_dof), self.P )
		# A_pos = np.kron(np.identity(self.num_dof), self.Pdot )
		# A_vel = np.kron(np.identity(self.num_dof), self.Pddot )
        
        # This is valid while not using knots and bernstein polynomials; directlly using velocity
		A_intpos = np.kron(np.identity(self.num_dof), self.Pint )
		A_pos = np.kron(np.identity(self.num_dof), self.P )
		A_vel = np.kron(np.identity(self.num_dof), self.Pdot )
		A_acc = np.kron(np.identity(self.num_dof), self.Pddot )

		return A_intpos, A_pos, A_vel, A_acc	


	def get_A_int_pos(self):
		A_int_pos = np.vstack(( self.Pint, -self.Pint))
		A_int_pos_ineq = np.kron(np.identity(self.num_dof), A_int_pos )
		return A_int_pos_ineq, A_int_pos
	
	def get_A_pos(self):
		A_pos = np.vstack(( self.P, -self.P     ))
		A_pos_ineq = np.kron(np.identity(self.num_dof), A_pos )
		return A_pos_ineq, A_pos

	def get_A_vel(self):
		A_vel = np.vstack(( self.Pdot, -self.Pdot  ))
		A_vel_ineq = np.kron(np.identity(self.num_dof), A_vel )
		return A_vel_ineq, A_vel
	
	def get_A_acc(self):
		A_acc = np.vstack(( self.Pddot, -self.Pddot  ))
		A_acc_ineq = np.kron(np.identity(self.num_dof), A_acc )
		return A_acc_ineq, A_acc
	
	def get_A_eq(self):
		return np.kron(np.identity(self.num_dof), self.P[0])

	
	@partial(jax.jit, static_argnums=(0,))
	def compute_boundary_vec_single(self, state_term):
		
		num_eq_constraint = 1 #int(jnp.shape(state_term)[0])
		b_eq_term = state_term.reshape( num_eq_constraint,self.num_dof).T
		b_eq_term = b_eq_term.reshape(num_eq_constraint*self.num_dof)

		return b_eq_term
	

	@partial(jax.jit, static_argnums=(0,))
	def compute_feasible_control(self, lamda_init, s_init, 
										 b_eq_term, xi_samples, 
										 init_pos):
		
	
		# Expand limits across the batch dimension: (num_batch, num_dof)
		u_max_batch = jnp.tile(jnp.expand_dims(self.p_max, axis=0), (self.num_batch, 1))
		u_min_batch = jnp.tile(jnp.expand_dims(self.p_min, axis=0), (self.num_batch, 1))
		init_pos_batch = jnp.tile(init_pos, (self.num_batch, 1))

		# Upper bounds: u_max - u_init (relative change limit)
		b_pos_upper = u_max_batch - init_pos_batch  # shape (num_batch, num_dof)

		# Lower bounds: -(u_min - u_init) for the -u constraint
		b_pos_lower = -(u_min_batch - init_pos_batch)  # shape (num_batch, num_dof)
		
		# Repeat across all time steps
		repeat_factor = self.num_control_constraints // (2 * self.num_dof) 
		
		b_pos_upper_flat = jnp.repeat(b_pos_upper, repeats=repeat_factor, axis=1) 
		b_pos_lower_flat = jnp.repeat(b_pos_lower, repeats=repeat_factor, axis=1)  
		
		b_control = jnp.hstack([b_pos_upper_flat, b_pos_lower_flat])

		# b_control = jnp.hstack((
		# 	self.pos_max * jnp.ones((self.num_batch, self.num_control_constraints // 2)),
		# 	self.pos_max * jnp.ones((self.num_batch, self.num_control_constraints // 2))
		# ))

		b_dcontrol = jnp.hstack((
			self.vel_max * jnp.ones((self.num_batch, self.num_dcontrol_constraints // 2)),
			self.vel_max * jnp.ones((self.num_batch, self.num_dcontrol_constraints // 2))
		))

		b_ddcontrol = jnp.hstack((
			self.acc_max * jnp.ones((self.num_batch, self.num_ddcontrol_constraints // 2)),
			self.acc_max * jnp.ones((self.num_batch, self.num_ddcontrol_constraints // 2))
		))
        


		# init_pos_batch = jnp.tile(init_pos, (self.num_batch, 1))  # (num_batch, 1)
        
		# # Calculate bounds for each joint and each batch
    	# # # Upper bounds: p_max - init_pos, Lower bounds: p_max + init_pos (assuming symmetric limits)
		# # b_pos_upper = (self.intcontrol_max - init_pos_batch)  # shape (num_batch, 1)
		# # b_pos_lower = (self.intcontrol_max + init_pos_batch)  # shape (num_batch, 1)
		# # #b_pos_lower = (-self.p_min + init_pos_batch)  # shape (num_batch, 1)
        
		
		# # # Corrected version
		# # b_pos_upper_flat = jnp.repeat(b_pos_upper, repeats=self.num_intcontrol_constraints // (2 * self.num_dof), axis=1) 
		# # b_pos_lower_flat = jnp.repeat(b_pos_lower, repeats=self.num_intcontrol_constraints // (2 * self.num_dof), axis=1)  
		# # b_pos = jnp.hstack([b_pos_upper_flat, b_pos_lower_flat])  


		# p_min = jnp.array([-20.0, -150.0, -45.0, -20.0, -150.0, -45.0])*jnp.pi/180
		# p_max = jnp.array([100.0, 0.0, 45.0, 100.0, 0.0, 45.0])*jnp.pi/180


		# # -----------------------
		# # Compute bounds per joint and batch
		# # -----------------------
		# # Expand limits across batch dimension
		# p_max_batch = jnp.tile(jnp.expand_dims(p_max, axis=0), (self.num_batch, 1))
		# p_min_batch = jnp.tile(jnp.expand_dims(p_min, axis=0), (self.num_batch, 1))

		# # Upper and lower bounds per joint
		# b_pos_upper = p_max_batch - init_pos_batch  # shape (num_batch, num_dof)
		# b_pos_lower = -p_min_batch + init_pos_batch # shape (num_batch, num_dof)

		# # Repeat across time / constraints
		# repeat_factor = self.num_intcontrol_constraints // (2 * self.num_dof)
		# b_pos_upper_flat = jnp.repeat(b_pos_upper, repeats=repeat_factor, axis=1)
		# b_pos_lower_flat = jnp.repeat(b_pos_lower, repeats=repeat_factor, axis=1)

		# # Final bounds: concatenate upper and lower bounds
		# # b_pos = jnp.hstack([b_pos_upper_flat, b_pos_lower_flat])
        
		# # b_control = jnp.hstack((b_control, b_dcontrol, b_ddcontrol, b_pos))

		b_control_all = jnp.hstack((b_control, b_dcontrol, b_ddcontrol))

		# Augmented bounds with slack variables
		b_control_aug = b_control_all - s_init


		# Cost matrix
		cost = (
			jnp.dot(self.A_projection.T, self.A_projection) +
			self.rho_ineq * jnp.dot(self.A_control.T, self.A_control)
		)

		# Linear cost term
		lincost = (
			-lamda_init -
			jnp.dot(self.A_projection.T, xi_samples.T).T -
			self.rho_ineq * jnp.dot(self.A_control.T, b_control_aug.T).T
		)

		# KKT system matrix
		cost_mat = jnp.vstack((
			jnp.hstack((cost, self.A_eq.T)),
			jnp.hstack((self.A_eq, jnp.zeros((self.A_eq.shape[0], self.A_eq.shape[0]))))
		))

		
		# Solve KKT system
		sol = jnp.linalg.solve(cost_mat, jnp.hstack((-lincost, b_eq_term)).T).T

		# Extract primal solution
		xi_projected = sol[:, :self.nvar]

		# Update slack variables
		s = jnp.maximum(
			jnp.zeros((self.num_batch, self.num_total_constraints)),
			-jnp.dot(self.A_control, xi_projected.T).T + b_control_all
		)

		# Compute residual
		res_vec = jnp.dot(self.A_control, xi_projected.T).T - b_control_all + s
		res_norm = jnp.linalg.norm(res_vec, axis=1)
		
		lamda = lamda_init - self.rho_ineq * jnp.dot(self.A_control.T, res_vec.T).T



		# lamda = lamda_init - self.rho_ineq * jnp.dot(self.A_control.T, res_vec.T).T - mu*g_grads_filt

		return xi_projected, s, res_norm, lamda
	

	@partial(jax.jit, static_argnums=(0,))
	def compute_projection(self, xi_samples, state_term, lamda_init, 
						   s_init, init_pos):
		
		b_eq_term = self.compute_boundary_vec_batch(state_term)  

		xi_projected_init = xi_samples

		def lax_custom_projection(carry, idx):
			_, lamda, s = carry
			lamda_prev, s_prev = lamda, s
			
			primal_sol, s, res_projection, lamda = self.compute_feasible_control(lamda, 
																		s, b_eq_term, xi_samples, 
																		init_pos)
			
			primal_residual = res_projection
			fixed_point_residual = (
				jnp.linalg.norm(lamda_prev - lamda, axis=1) +
				jnp.linalg.norm(s_prev - s, axis=1)
			)
			return (primal_sol, lamda, s), (primal_residual, fixed_point_residual)

		carry_init = (xi_projected_init, lamda_init, s_init)


		carry_final, res_tot = jax.lax.scan(
			lax_custom_projection,
			carry_init,
			jnp.arange(self.maxiter_projection)
		)

		primal_sol, lamda, s = carry_final
		primal_residuals, fixed_point_residuals = res_tot

		primal_residuals = jnp.stack(primal_residuals)
		fixed_point_residuals = jnp.stack(fixed_point_residuals)

		return primal_sol, primal_residuals, fixed_point_residuals

	
	@partial(jax.jit, static_argnums=(0,))
	def mjx_step_control(self, mjx_data, control_single):
		
		# Apply the controls using the 'ctrl' field of mjx_data.
		# We assume 'control_single' contains the controls for the joints
		# corresponding to 'self.joint_mask_ctrl' (or similar mask for control inputs).
		# NOTE: MuJoCo's control inputs are typically applied to 'mjx_data.ctrl'.
		#ctrl = mjx_data.ctrl.at[self.joint_mask_ctrl].set(control_single)

        # self.data.ctrl[self.actuator_ctrl_indices] = self.control
		ctrl = mjx_data.ctrl.at[jnp.array(self.actuator_ctrl_indices)].set(control_single)

		# ctrl = mjx_data.ctrl.at[self.actuator_ctrl_indices].set(control_single)
		# ctrl = mjx_data.ctrl.at[self.joint_ctrl_indices].set(control_single)
		mjx_data = mjx_data.replace(ctrl=ctrl)
		
		# Step the simulation
		mjx_data = self.jit_step(self.mjx_model, mjx_data)

		# Get joint positions and end-effector states
		theta = mjx_data.qpos[self.joint_mask_pos]

		torso_pos = mjx_data.site_xpos[self.torso_id]
		
		# Collision detection
		# collision = mjx_data.contact.dist[self.mask]
		collision = mjx_data.contact.dist

		# Extract ONLY the specific sensor data you need for cost computation

		# torso_height =  sitexpos[self.torso_id, 2] torso_pos[2]  # z-coordinate of torso

		torso_height = mjx_data.site_xpos[self.torso_id, 2]
		sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
		torso_orientation = mjx_data.sensordata[sensor_adr : sensor_adr + 4]  # quaternion
		
		# Get joint positions for nominal cost
		joint_positions = mjx_data.qpos[7:]  # Adjust index based on your robot structure

		return mjx_data, (
			theta, 
			torso_pos,
			collision,
			torso_height,
			torso_orientation,
			joint_positions
		)


    
	@partial(jax.jit, static_argnums=(0,))
	def compute_rollout_single_control(self, mjx_data_current, controls, init_pos, init_vel):
		# Initialize MJX data with initial position and velocity
		# mjx_data_current = self.mjx_data
		# mjx_data = self.mjx_data
		qvel = mjx_data_current.qvel.at[self.joint_mask_vel].set(init_vel)
		qpos = mjx_data_current.qpos.at[self.joint_mask_pos].set(init_pos)

		mjx_data = mjx_data_current.replace(qvel=qvel, qpos=qpos)

		# 2. Reshape the input controls for jax.lax.scan
		# The input is now 'controls' (a control signal) instead of 'thetadot'.
		# Note: 'self.num_dof' should likely be replaced by 'self.num_ctrl' or
		# the dimension of the control space if they are different.
		control_single = controls.reshape(self.num_dof, self.num)

		
		# 3. Perform the rollout using the control-based step function
		# Call self.mjx_step_control recursively
		def step_fn(carry, control):
			mjx_data = carry
			# Use your existing mjx_step_torque function
			mjx_data_next, (theta, torso_pos, collision, 
				            torso_height, torso_orientation, joint_positions) = self.mjx_step_control(mjx_data, control)
	
			return mjx_data_next, (theta, torso_pos, collision, torso_height, torso_orientation, joint_positions)
		
		#mjx_data_final, out = jax.lax.scan(self.mjx_step_control, mjx_data, control_single.T, length=self.num)
		
		# Perform the rollout using scan to collect all states and sensor data
		mjx_data_final, out = jax.lax.scan(step_fn, mjx_data, control_single.T, length=self.num)
		
		# theta, torso_pos, collision, site_xpos_all, sensor_data_all = out

		theta, torso_pos, collision, torso_height_all, torso_orientation_all, joint_positions_all = out

		# sensor_data = mjx_data_final.sensordata
		
		return theta.T.flatten(), torso_pos, collision, torso_height_all, torso_orientation_all, joint_positions_all


	

	# @partial(jax.jit, static_argnums=(0,))
	# def compute_rollout_single(self, mjx_data_current, thetadot, init_pos, init_vel):
	# 	# Use the passed-in current state instead of self.mjx_data
	# 	# mjx_data_current = self.mjx_data
	# 	qvel = mjx_data_current.qvel.at[self.joint_mask_vel].set(init_vel)
	# 	qpos = mjx_data_current.qpos.at[self.joint_mask_pos].set(init_pos)
		
	# 	mjx_data = mjx_data_current.replace(qvel=qvel, qpos=qpos)
		
	# 	thetadot_single = thetadot.reshape(self.num_dof, self.num)
	# 	mjx_data_final, out = jax.lax.scan(self.mjx_step, mjx_data, thetadot_single.T, length=self.num)
	# 	theta, torso_pos, collision = out
	# 	sensor_data = mjx_data_final.sensordata
		
	# 	return theta.T.flatten(), torso_pos, collision, sensor_data



	@partial(jax.jit, static_argnums=(0,))
	def compute_cost_single(self, control_single, torso_height_full_horizon, torso_orientation_full_horizon, joint_positions_full_horizon,  cost_weights):
	    
		def _get_torso_height(sitexpos) -> jax.Array:
			"""Get the height of the torso above the ground."""
			return sitexpos[self.torso_id, 2]
		
		def _get_torso_orientation(quat) -> jax.Array:
			"""Get the rotation from the current torso orientation to upright."""
			# sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
			# quat = sensordata[sensor_adr : sensor_adr + 4]
			upright = jnp.array([0.0, 0.0, 1.0])
			return mjx._src.math.rotate(upright, quat)
		
		H = jnp.arange(self.num)
		discounts = self.gamma ** H
		
		# Vectorize over time dimension (num_steps)
		get_torso_orientation_batched_over_horizon = jax.vmap(_get_torso_orientation, in_axes=(0,))
		
		# Apply rotation for all timesteps → shape (num_steps, 3)
		torso_dirs = get_torso_orientation_batched_over_horizon(torso_orientation_full_horizon)

		orientation_cost = jnp.sum(
            # jnp.square(_get_torso_orientation(mjx_data))
            discounts[:, None] * jnp.square(torso_dirs[:,:2])
        )
        
		#→ shape (num_steps,)
		height_cost = jnp.sum(discounts* jnp.square(torso_height_full_horizon-self.target_height))
        
		#→ shape (num_steps, num_dof)
		# nominal_cost = jnp.sum(jnp.square(mjx_data.qpos[7:] - self.qstand[7:]))
		nominal_cost = jnp.sum(discounts[:, None] * jnp.square(joint_positions_full_horizon - self.qstand[7:]))



		cost = (
			cost_weights['orientation'] * orientation_cost 
			+cost_weights['height'] * height_cost 
			+ cost_weights['nominal'] * nominal_cost
		)	

		cost_list = jnp.array([
			cost_weights['orientation'] * orientation_cost,
			cost_weights['height'] * height_cost, 
			cost_weights['nominal'] * nominal_cost
		])

		return cost, cost_list
	
	@partial(jax.jit, static_argnums=(0, ))
	def compute_ellite_samples(self, cost_batch, xi_filtered):
		idx_ellite = jnp.argsort(cost_batch)
		cost_ellite = cost_batch[idx_ellite[0:self.ellite_num]]
		xi_ellite = xi_filtered[idx_ellite[0:self.ellite_num]]
		return xi_ellite, idx_ellite, cost_ellite
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_xi_samples(self, key, xi_mean, xi_cov ):
		key, subkey = jax.random.split(key)
		xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov, (self.num_batch, ))
		# xi_samples = jnp.clip(xi_samples, a_min =-1.0, a_max=1.0)
		return xi_samples, key
	
	@partial(jax.jit, static_argnums=(0,))
	def comp_prod(self, diffs, d ):
		term_1 = jnp.expand_dims(diffs, axis = 1)
		term_2 = jnp.expand_dims(diffs, axis = 0)
		prods = d * jnp.outer(term_1,term_2)
		return prods	
	
	@partial(jax.jit, static_argnums=(0,))
	def repair_cov(self, C):
		epsilon = 1e-5
		eigenvalues, eigenvectors = jnp.linalg.eigh(C)
		min_eigenvalue = jnp.min(eigenvalues)
		def repair(_):
			clipped = jnp.where(eigenvalues < epsilon, epsilon, eigenvalues)
			D_prime = jnp.diag(clipped)
			C_repaired = eigenvectors @ D_prime @ eigenvectors.T
			# C_repaired = (C_repaired + C_repaired.T) / 2
			return C_repaired

		def keep(_):
			# cov_sym = (cov + cov.T) / 2
			return C

		C_repaired = jax.lax.cond(min_eigenvalue < epsilon, repair, keep, operand=None)
		return C_repaired
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_mean_cov(self, cost_ellite, mean_control_prev, cov_control_prev, xi_ellite):
		w = cost_ellite
		w_min = jnp.min(cost_ellite)
		w = jnp.exp(-(1/self.lamda) * (w - w_min ) )
		sum_w = jnp.sum(w, axis = 0)
		mean_control = (1-self.alpha_mean)*mean_control_prev + self.alpha_mean*(jnp.sum( (xi_ellite * w[:,jnp.newaxis]) , axis= 0)/ sum_w)
		diffs = (xi_ellite - mean_control)
		prod_result = self.vec_product(diffs, w)
		cov_control = (1-self.alpha_cov)*cov_control_prev + self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.2*jnp.identity(self.nvar)
		cov_control = self.repair_cov(cov_control)
		return mean_control, cov_control
	
	@partial(jax.jit, static_argnums=(0,))
	def cem_iter(self, carry,  scan_over):

		xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples, init_pos, init_vel, cost_weights, mjx_data_current = carry

		xi_mean_prev = xi_mean 
		xi_cov_prev = xi_cov

		# xi_samples_reshaped = xi_samples.reshape(self.num_batch, self.num_dof, self.num)
		# xi_samples_batched_over_dof = jnp.transpose(xi_samples_reshaped, (1, 0, 2)) # shape: (DoF, B, num)

		# state_term_reshaped = state_term.reshape(self.num_batch, self.num_dof, 1)
		# state_term_batched_over_dof = jnp.transpose(state_term_reshaped, (1, 0, 2)) #Shape: (DoF, B, 1)

		# lamda_init_reshaped = lamda_init.reshape(self.num_batch, self.num_dof, self.num)
		# lamda_init_batched_over_dof = jnp.transpose(lamda_init_reshaped, (1, 0, 2)) # shape: (DoF, B, num)

		# s_init_reshaped = s_init.reshape(self.num_batch, self.num_dof, self.num_total_constraints_per_dof )
		# s_init_batched_over_dof = jnp.transpose(s_init_reshaped, (1, 0, 2)) # shape: (DoF, B, num_total_constraints_per_dof)


		
        # Pass all arguments as positional arguments; not keyword arguments
		xi_filtered, primal_residuals, fixed_point_residuals = self.compute_projection(
			                                                     xi_samples, 
														         state_term, 
																 lamda_init, 
																 s_init, 
																 init_pos)
		
		# xi_filtered = xi_filtered.transpose(1, 0, 2).reshape(self.num_batch, -1) # shape: (B, num*num_dof)
		
		# primal_residuals = jnp.linalg.norm(primal_residuals, axis = 0)
		# fixed_point_residuals = jnp.linalg.norm(fixed_point_residuals, axis = 0)
				
		avg_res_primal = jnp.sum(primal_residuals, axis = 0)/self.maxiter_projection
    	
		avg_res_fixed_point = jnp.sum(fixed_point_residuals, axis = 0)/self.maxiter_projection

		
		# control = jnp.dot(self.A_pos, xi_samples.T).T
		control = jnp.dot(self.A_pos, xi_filtered.T).T

		mjx_data_current = carry[-1]

		(theta, torso_pos, 
		collision, torso_height_all, 
		torso_orientation_all, 
		joint_positions_all) = self.compute_rollout_batch(mjx_data_current, control, init_pos, init_vel)

		# print("torso_height_all", np.shape(torso_height_all)) # Should be (num_batch, num_steps)
		# print("torso_orientation_all", np.shape(torso_orientation_all)) # Should be (num_batch, num_steps, 4)
		# print("joint_positions_all", np.shape(joint_positions_all)) # Should be (num_batch, num_steps, num_dof)


		cost_batch, cost_list_batch = self.compute_cost_batch(control, torso_height_all, 
														torso_orientation_all, joint_positions_all, cost_weights)

		xi_ellite, idx_ellite, cost_ellite = self.compute_ellite_samples(cost_batch, xi_samples)
		xi_mean, xi_cov = self.compute_mean_cov(cost_ellite, xi_mean_prev, xi_cov_prev, xi_ellite)
		xi_samples_new, key = self.compute_xi_samples(key, xi_mean, xi_cov)

		carry = (xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples_new, init_pos, init_vel, cost_weights, mjx_data_current)

		return carry, (cost_batch, cost_list_batch, control, theta, xi_samples,
				 avg_res_primal, avg_res_fixed_point, primal_residuals, fixed_point_residuals, xi_filtered, torso_pos)
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_cem(
		self, current_mjx_data, xi_mean, 
		xi_cov,
		init_pos, 
		init_vel, 
		init_acc,
		init_ctrl,
		lamda_init,
		s_init,
		xi_samples,
		cost_weights,
		):


		ctrl_init = jnp.tile(init_ctrl, (self.num_batch, 1))

		state_term = ctrl_init	
		
		key, subkey = jax.random.split(self.key)

		carry = (xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples, init_pos, init_vel, cost_weights, current_mjx_data)
		scan_over = jnp.array([0]*self.maxiter_cem)
		
		carry, out = jax.lax.scan(self.cem_iter, carry, scan_over, length=self.maxiter_cem)

		(cost_batch, cost_list_batch, control, theta, 
         xi_samples_all, avg_res_primal, avg_res_fixed, 
		 primal_residuals, fixed_point_residuals, xi_filtered, torso_pos) = out

		idx_min = jnp.argmin(cost_batch[-1])
		cost = jnp.min(cost_batch, axis=1)
		best_controls = control[-1][idx_min].reshape((self.num_dof, self.num)).T
		best_traj = theta[-1][idx_min].reshape((self.num_dof, self.num)).T

		best_cost_list = cost_list_batch[-1][idx_min]

		best_cost = cost_batch[-1][idx_min]

		best_cost_list_cem = cost_list_batch[:, idx_min]

		best_cost_cem = cost_batch[:, idx_min]

		xi_mean = carry[0]
		xi_cov = carry[1]


		torso_pos_planned = torso_pos[-1][idx_min]

	    
		return (
			best_cost_cem,
			best_cost_list_cem,
			best_controls,
			best_traj,
			xi_mean,
			xi_cov,
			xi_samples_all,
			xi_filtered,
			control,
			theta,
			avg_res_primal,
			avg_res_fixed,
			primal_residuals,
			fixed_point_residuals,
			idx_min,
			torso_pos_planned,
			torso_pos
		)