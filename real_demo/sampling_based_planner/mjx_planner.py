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
			     maxiter_projection=None, max_joint_intforce = None ,max_joint_force = None, 
				 max_joint_dforce = None, max_joint_ddforce = None):
		super(cem_planner, self).__init__()
	    

		self.num_dof = num_dof
		self.num_batch = num_batch
		self.t = timestep
		self.num = num_steps
		self.num_elite = num_elite

		self.t_fin = self.num*self.t
		# self.init_joint_position = np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])
		self.init_joint_position = np.array([0.0])
		
		tot_time = np.linspace(0, self.t_fin, self.num)
		self.tot_time = tot_time
		tot_time_copy = tot_time.reshape(self.num, 1)
     
		# self.P = jnp.identity(self.num) # Velocity mapping 
		# self.Pdot = jnp.diff(self.P, axis=0)/self.t # Accelaration mapping
		# self.Pddot = jnp.diff(self.Pdot, axis=0)/self.t # Jerk mapping
		# self.Pint = jnp.cumsum(self.P, axis=0)*self.t # Position mapping
		
		# self.P, self.Pdot, self.Pddot = bernstein_coeff_ordern_new(self.num-1, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
		self.P, self.Pdot, self.Pddot = bernstein_coeff_ordern_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

		self.Pint = jnp.zeros_like(self.P)
	
		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
		self.Pint_jax = jnp.asarray(self.Pint)

		self.nvar_single = jnp.shape(self.P_jax)[1]
		self.nvar = self.nvar_single*self.num_dof 
  
		self.rho_ineq = 5.0
		self.rho_projection = 1.0

		self.A_projection = jnp.identity(self.nvar)

		A_force_ineq, A_force = self.get_A_force()
		self.A_force_ineq = jnp.asarray(A_force_ineq) 
		self.A_force = jnp.asarray(A_force)

		A_dforce_ineq, A_dforce = self.get_A_dforce()
		self.A_dforce_ineq = jnp.asarray(A_dforce_ineq) 
		self.A_dforce = jnp.asarray(A_dforce)

		A_ddforce_ineq, A_ddforce = self.get_A_ddforce()
		self.A_ddforce_ineq = jnp.asarray(A_ddforce_ineq)
		self.A_ddforce = jnp.asarray(A_ddforce)
  
		A_int_force_ineq, A_int_force = self.get_A_int_force()
		self.A_int_force_ineq = jnp.asarray(A_int_force_ineq) 
		self.A_int_force = jnp.asarray(A_int_force)

		# Combined control matrix (like A_control in )
		self.A_control = jnp.vstack((
			self.A_force_ineq,
			self.A_dforce_ineq,
			self.A_ddforce_ineq
		#	self.A_int_force_ineq
		))

		A_eq = self.get_A_eq()
		self.A_eq = jnp.asarray(A_eq)

		A_intforce, A_force, A_dforce, A_ddforce = self.get_A_force_control()

		self.A_intforce = np.asarray(A_intforce)
		self.A_force = np.asarray(A_force)
		self.A_dforce = np.asarray(A_dforce)
		self.A_ddforce = np.asarray(A_ddforce)
		
		self.key= jax.random.PRNGKey(42)
		self.maxiter_projection = maxiter_projection
		self.maxiter_cem = maxiter_cem

		self.force_max = max_joint_force
		self.dforce_max = max_joint_dforce
		self.ddforce_max = max_joint_ddforce
		# self.intforce_max = max_joint_intforce		

				
		
		    
    	# Calculating number of Inequality constraints
		# self.num_force = self.num
		# self.num_dforce = self.num - 1
		# self.num_ddforce = self.num - 2
		# self.num_intforce = self.num

		self.num_force   = self.P.shape[0]       # number of time samples for force 
		self.num_dforce  = self.Pdot.shape[0]    # number of samples for rate of change)
		self.num_ddforce = self.Pddot.shape[0]   # number of samples for double rate of change
		self.num_intforce = self.Pint.shape[0]   # number of samples for integrated force

		self.num_force_constraints = 2 * self.num_force * num_dof
		self.num_dforce_constraints = 2 * self.num_dforce * num_dof
		self.num_ddforce_constraints = 2 * self.num_ddforce * num_dof
		self.num_intforce_constraints = 2 * self.num_intforce * num_dof
		self.num_total_constraints = (self.num_force_constraints + self.num_dforce_constraints + 
								      self.num_ddforce_constraints)
		self.num_total_constraints_per_dof = self.num_total_constraints / self.num_dof



		self.b_force = jnp.hstack((
			self.force_max * jnp.ones((self.num_batch, self.num_force_constraints // 2)),
			self.force_max * jnp.ones((self.num_batch, self.num_force_constraints // 2))
		))

		self.b_dforce = jnp.hstack((
			self.dforce_max * jnp.ones((self.num_batch, self.num_dforce_constraints // 2)),
			self.dforce_max * jnp.ones((self.num_batch, self.num_dforce_constraints // 2))
		))

		self.b_ddforce = jnp.hstack((
			self.ddforce_max * jnp.ones((self.num_batch, self.num_ddforce_constraints // 2)),
			self.ddforce_max * jnp.ones((self.num_batch, self.num_ddforce_constraints // 2))
		))
        
        

		self.b_control = jnp.hstack((self.b_force, self.b_dforce, self.b_ddforce))

		self.ellite_num = int(self.num_elite*self.num_batch)



		self.alpha_mean = 0.6
		self.alpha_cov = 0.6

		self.lamda = 0.1
		self.g = 10
		self.vec_product = jax.jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

		self.gamma = 0.99 #Discount factor for reward

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

	
		# robot_joints = np.array(['pendulum_joint'])
		robot_joints = np.array(['slider']) #Only actuated joint should be mentioned here
		
		self.joint_mask_pos = np.isin(np.array(joint_names_pos), robot_joints)
		self.joint_mask_vel = np.isin(np.array(joint_names_vel), robot_joints)
		self.joint_mask_ctrl = np.isin(np.array(joint_names_ctrl), robot_joints)
		self.joint_ctrl_indices = jnp.where(self.joint_mask_ctrl)[0]
		actuator_joint_ids = self.model.actuator_trnid[:, 0]
		self.actuator_ctrl_indices = [
			i for i, j in enumerate(actuator_joint_ids)
			if self.joint_mask_ctrl[j]
		]

		# # Add this after your existing joint mask code in __init__
		# print("\n=== DEBUG: JOINT MASKS ===")
		# print("Position-controlled joints:")
		# for i, name in enumerate(joint_names_pos):
		# 	print(f"  {i}: '{name}' -> controlled: {self.joint_mask_pos[i]}")

		print("\ force-controlled joints:")  
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



		self.tip = self.model.site(name="tip").id


		self.compute_rollout_batch = jax.vmap(self.compute_rollout_single_force, in_axes = (None, 0, None, None))
		self.compute_cost_batch = jax.vmap(self.compute_cost_single, in_axes = (0, 0, 0, None))
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
			# f'\n Number of geomteric IDs for colllision: {len(self.geom_ids_all)}'
		    # f'\n{self.mask.sum()} / {self.mask.shape[0]} contacts involve robot.'
		)
    
	def angle_normalize(self, x):
		return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    
	def get_A_force_control(self):

		# This is valid while dealing with knots anfd projecting into pos,vel,acc space with Bernstein Polynomials
		# A_intforce = np.kron(np.identity(self.num_dof), self.P )
		# A_force = np.kron(np.identity(self.num_dof), self.Pdot )
		# A_dforce = np.kron(np.identity(self.num_dof), self.Pddot )
        
        # This is valid while not using knots and bernstein polynomials; directlly using velocity
		A_intforce = np.kron(np.identity(self.num_dof), self.Pint )
		A_force = np.kron(np.identity(self.num_dof), self.P )
		A_dforce = np.kron(np.identity(self.num_dof), self.Pdot )
		A_ddforce = np.kron(np.identity(self.num_dof), self.Pddot )

		return A_intforce, A_force, A_dforce, A_ddforce	


	def get_A_int_force(self):
		A_int_force = np.vstack(( self.Pint, -self.Pint))
		A_int_force_ineq = np.kron(np.identity(self.num_dof), A_int_force )
		return A_int_force_ineq, A_int_force
	
	def get_A_force(self):
		A_force = np.vstack(( self.P, -self.P ))
		A_force_ineq = np.kron(np.identity(self.num_dof), A_force )
		return A_force_ineq, A_force

	def get_A_dforce(self):
		A_dforce = np.vstack(( self.Pdot, -self.Pdot  ))
		A_dforce_ineq = np.kron(np.identity(self.num_dof), A_dforce )
		return A_dforce_ineq, A_dforce
	
	def get_A_ddforce(self):
		A_ddforce = np.vstack(( self.Pddot, -self.Pddot  ))
		A_ddforce_ineq = np.kron(np.identity(self.num_dof), A_ddforce )
		return A_ddforce_ineq, A_ddforce
	
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
		
	


		# Augmented bounds with slack variables
		b_control_aug = self.b_control - s_init


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
			-jnp.dot(self.A_control, xi_projected.T).T + self.b_control
		)

		# Compute residual
		res_vec = jnp.dot(self.A_control, xi_projected.T).T - self.b_control + s
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

	

	# @partial(jax.jit, static_argnums=(0,))
	# def mjx_step(self, mjx_data, thetadot_single):
	
	# 	qvel = mjx_data.qvel.at[self.joint_mask_vel].set(thetadot_single)
	# 	mjx_data = mjx_data.replace(qvel=qvel)
		
	# 	# Step the simulation
	# 	mjx_data = self.jit_step(self.mjx_model, mjx_data)

	# 	# Get joint positions and end-effector states
	# 	theta = mjx_data.qpos[self.joint_mask_pos]

	# 	torso_pos = mjx_data.site_xpos[self.torso]
		

		
	# 	# Collision detection
	# 	# collision = mjx_data.contact.dist[self.mask]
	# 	collision = mjx_data.contact.dist

	# 	return mjx_data, (
	# 		theta, 
	# 		torso_pos,
	# 		collision
	# 	)


	@partial(jax.jit, static_argnums=(0,))
	def mjx_step_force(self, mjx_data, force_single):
		
		# Apply the forces using the 'ctrl' field of mjx_data.
		# We assume 'force_single' contains the forces for the joints
		# corresponding to 'self.joint_mask_ctrl' (or similar mask for control inputs).
		# NOTE: MuJoCo's control inputs are typically applied to 'mjx_data.ctrl'.
		#ctrl = mjx_data.ctrl.at[self.joint_mask_ctrl].set(force_single)

        # self.data.ctrl[self.actuator_ctrl_indices] = self.force
		ctrl = mjx_data.ctrl.at[jnp.array(self.actuator_ctrl_indices)].set(force_single)

		# ctrl = mjx_data.ctrl.at[self.actuator_ctrl_indices].set(force_single)
		# ctrl = mjx_data.ctrl.at[self.joint_ctrl_indices].set(force_single)
		mjx_data = mjx_data.replace(ctrl=ctrl)
		
		# Step the simulation
		mjx_data = self.jit_step(self.mjx_model, mjx_data)

		# Get joint positions and end-effector states
		theta = mjx_data.qpos[1]  #Joint indices Checked from hydrax and verified after lading joints from xml 
		thetadot = mjx_data.qvel[1]

		tip_pos = mjx_data.site_xpos[self.tip]
		
		# Collision detection
		# collision = mjx_data.contact.dist[self.mask]

		return mjx_data, (
			theta, 
			thetadot,
			tip_pos
		)
	


	# @partial(jax.jit, static_argnums=(0,))
	# def compute_rollout_single(self, thetadot, init_pos, init_vel):

	# 	mjx_data = self.mjx_data
	# 	qvel = mjx_data.qvel.at[self.joint_mask_vel].set(init_vel)
	# 	qpos = mjx_data.qpos.at[self.joint_mask_pos].set(init_pos)

	# 	mjx_data = mjx_data.replace(qvel=qvel, qpos=qpos)

	# 	thetadot_single = thetadot.reshape(self.num_dof, self.num)
	# 	mjx_data_final, out = jax.lax.scan(self.mjx_step, mjx_data, thetadot_single.T, length=self.num)
	# 	theta, torso_pos, collision = out
	# 	#Sensor data
	# 	sensor_data = mjx_data_final.sensordata

	# 	return theta.T.flatten(), torso_pos, collision, sensor_data

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
	def compute_rollout_single_force(self, mjx_data_current, forces, init_pos, init_vel):
		# Initialize MJX data with initial position and velocity
		# mjx_data_current = self.mjx_data
		# mjx_data = self.mjx_data
		qvel = mjx_data_current.qvel.at[self.joint_mask_vel].set(init_vel)
		qpos = mjx_data_current.qpos.at[self.joint_mask_pos].set(init_pos)

		mjx_data = mjx_data_current.replace(qvel=qvel, qpos=qpos)

		# 2. Reshape the input forces for jax.lax.scan
		# The input is now 'forces' (a control signal) instead of 'thetadot'.
		# Note: 'self.num_dof' should likely be replaced by 'self.num_ctrl' or
		# the dimension of the control space if they are different.
		force_single = forces.reshape(self.num_dof, self.num)
		
		# 3. Perform the rollout using the force-based step function
		# Call self.mjx_step_force instead of self.mjx_step.
		mjx_data_final, out = jax.lax.scan(self.mjx_step_force, mjx_data, force_single.T, length=self.num)
		
		theta, thetadot, tip_pos = out

		# sensor_data = mjx_data_final.sensordata
		
		return theta.T.flatten(), thetadot.T.flatten(), tip_pos
	


	@partial(jax.jit, static_argnums=(0,))
	def compute_cost_single(self, force_single, theta, thetadot, cost_weights):
	

		# # --- Inline sensor extraction ---
		# def get_torso_height(sensor_data) -> jax.Array:
		# 	"""Get the height of the torso above the ground."""
		# 	sensor_adr = self.model.sensor_adr[self.torso_position_sensor]
		# 	return sensor_data[sensor_adr + 2]  # px, py, pz

		H = jnp.arange(self.num)
		discounts = self.gamma ** H
		
		def _distance_to_upright(theta) -> jax.Array:
			"""Get a measure of distance to the upright position."""
			theta_ = theta + jnp.pi
			# jax.debug.print("theta {}", theta)
			# theta_err = self.angle_normalize(theta_)
			theta_err = jnp.array([jnp.cos(theta_) - 1, jnp.sin(theta_)])
			#jax.debug.print("theta_err {}", theta_err)
			return jnp.sum(discounts * jnp.square(theta_err))

		theta_cost = _distance_to_upright(theta)

		thetadot_cost = jnp.sum(discounts * jnp.square(thetadot))

		control_cost = jnp.sum(discounts * jnp.square(force_single))

		cost = (
			cost_weights['theta'] * theta_cost 
			+ cost_weights['thetadot'] * thetadot_cost 
			+ cost_weights['control'] * control_cost 
		)	

		cost_list = jnp.array([
			cost_weights['theta'] * theta_cost, 
			cost_weights['thetadot'] * theta_cost, 
			cost_weights['control'] * control_cost
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
		# xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov+0.009*jnp.identity(self.nvar), (self.num_batch, ))
		xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov, (self.num_batch, ))
		xi_samples = jnp.clip(xi_samples, a_min=-1.8, a_max=1.8)
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
		cov_control = (1-self.alpha_cov)*cov_control_prev + self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.01*jnp.identity(self.nvar)
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
        
		s_init = jnp.maximum(
			jnp.zeros((self.num_batch, self.num_total_constraints)),
			-jnp.dot(self.A_control, xi_samples.T).T + self.b_control
		)
		
        # Pass all arguments as positional arguments; not keyword arguments
		xi_filtered, primal_residuals, fixed_point_residuals = self.compute_projection(
			                                                     xi_samples, 
														         state_term, 
																 lamda_init, 
																 s_init, 
																 init_pos)
		

		xi_filtered = jnp.clip(xi_filtered, a_min=-1.0, a_max=1.0)
		

		# xi_filtered = xi_filtered.transpose(1, 0, 2).reshape(self.num_batch, -1) # shape: (B, num*num_dof)
		
		# primal_residuals = jnp.linalg.norm(primal_residuals, axis = 0)
		# fixed_point_residuals = jnp.linalg.norm(fixed_point_residuals, axis = 0)

				
		avg_res_primal = jnp.sum(primal_residuals, axis = 0)/self.maxiter_projection
    	
		avg_res_fixed_point = jnp.sum(fixed_point_residuals, axis = 0)/self.maxiter_projection

		# force = jnp.dot(self.A_force, xi_filtered.T).T
		
		force = jnp.dot(self.A_force, xi_samples.T).T

		# jax.debug.print("xi_samples {}", jnp.shape(xi_samples))
		# jax.debug.print("force {}", jnp.shape(force))

		mjx_data_current = carry[-1]

		theta, thetadot, tip_pos = self.compute_rollout_batch(mjx_data_current, force, init_pos, init_vel)
		cost_batch, cost_list_batch = self.compute_cost_batch(force, theta, thetadot, cost_weights)

		xi_ellite, idx_ellite, cost_ellite = self.compute_ellite_samples(cost_batch, xi_samples)
		xi_mean, xi_cov = self.compute_mean_cov(cost_ellite, xi_mean_prev, xi_cov_prev, xi_ellite)
		xi_samples_new, key = self.compute_xi_samples(key, xi_mean, xi_cov)

		carry = (xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples_new, init_pos, init_vel, cost_weights, mjx_data_current)

		return carry, (cost_batch, cost_list_batch, force, theta, 
				 avg_res_primal, avg_res_fixed_point, primal_residuals, fixed_point_residuals, xi_filtered, tip_pos)
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_cem(
		self, current_mjx_data, xi_mean, 
		xi_cov,
		init_pos, 
		init_vel, 
		init_acc,
		init_force,
		lamda_init,
		s_init,
		xi_samples,
		cost_weights,
		):


		force_init = jnp.tile(init_force, (self.num_batch, 1))

		state_term = force_init	
		
		key, subkey = jax.random.split(self.key)

		carry = (xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples, init_pos, init_force, cost_weights, current_mjx_data)
		scan_over = jnp.array([0]*self.maxiter_cem)
		
		carry, out = jax.lax.scan(self.cem_iter, carry, scan_over, length=self.maxiter_cem)
		cost_batch, cost_list_batch, force, theta, avg_res_primal, avg_res_fixed, primal_residuals, fixed_point_residuals, xi_filtered ,tip_pos = out

		idx_min = jnp.argmin(cost_batch[-1])
		cost = jnp.min(cost_batch, axis=1)
		best_forces = force[-1][idx_min].reshape((self.num_dof, self.num)).T
		best_traj = theta[-1][idx_min].reshape((self.num_dof, self.num)).T

		best_cost_list = cost_list_batch[-1][idx_min]

		best_cost = cost_batch[-1][idx_min]

		best_cost_list_cem = cost_list_batch[:, idx_min]

		best_cost_cem = cost_batch[:, idx_min]
		
		xi_mean = carry[0]
		xi_cov = carry[1]


		tip_pos_planned = tip_pos[-1][idx_min]

	    
		return (
			best_cost_cem,
			best_cost_list_cem,
			best_forces,
			best_traj,
			xi_mean,
			xi_cov,
			xi_filtered,
			force,
			theta,
			avg_res_primal,
			avg_res_fixed,
			primal_residuals,
			fixed_point_residuals,
			idx_min,
			tip_pos_planned,
			tip_pos
		)