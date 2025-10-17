from sampling_based_planner.mjx_planner import cem_planner
from sampling_based_planner.quat_math import quaternion_distance, quaternion_multiply, rotation_quaternion
from sampling_based_planner.Simple_MLP.mlp_singledof import MLP, MLPProjectionFilter
from ik_based_planner.ik_solver import InverseKinematicsSolver

import mujoco
from mujoco import viewer
import jax.numpy as jnp
import jax

import numpy as np
import torch 
import contextlib
from io import StringIO



class run_cem_planner:
    def __init__(self, model, data, num_dof=12, num_batch=500, num_steps=20, 
                 maxiter_cem=1, maxiter_projection=5, num_elite=0.05, timestep=None,
                 max_joint_inttorque=0.0, max_joint_torque=1.0, 
                 max_joint_dtorque=1.5, max_joint_ddtorque=2.0,
                 device='cuda', cost_weights=None):
        
        # Initialize parameters
        self.model = model
        self.data = data
        self.num_dof = num_dof
        self.num_batch = num_batch
        self.num_steps = num_steps
        self.maxiter_cem = maxiter_cem
        self.maxiter_projection = maxiter_projection
        self.num_elite = num_elite
        self.timestep = timestep
        self.device = device

        self.cost_weights = cost_weights

        # Initialize CEM planner
        self.cem = cem_planner(
            model=model,
            num_dof=num_dof, 
            num_batch=num_batch, 
            num_steps=num_steps, 
            maxiter_cem=maxiter_cem,
            num_elite=num_elite,
            timestep=timestep,
            maxiter_projection=maxiter_projection,
            max_joint_inttorque=max_joint_inttorque,
            max_joint_torque=max_joint_torque,
            max_joint_dtorque=max_joint_dtorque,
            max_joint_ddtorque=max_joint_ddtorque
        )
        
        # Initialize CEM variables
        self.cov_coeff_scalar = 0.5
        self.xi_mean_single = jnp.zeros(self.cem.nvar_single)
        self.xi_cov_single = self.cov_coeff_scalar*jnp.identity(self.cem.nvar_single)
        self.xi_mean = jnp.tile(self.xi_mean_single, self.cem.num_dof)
        self.xi_cov = jnp.kron(jnp.eye(self.cem.num_dof), self.xi_cov_single)
        self.lamda_init = jnp.zeros((num_batch, self.cem.nvar))
        self.s_init = jnp.zeros((num_batch, self.cem.num_total_constraints))
        self.key = jax.random.PRNGKey(0)
        

        
        # # Get TCP references for both arms
        # self.tcp_id_0 = model.site(name="tcp_0").id
        # self.hande_id_0 = model.body(name="hande_0").id
        # self.tcp_id_1 = model.site(name="tcp_1").id
        # self.hande_id_1 = model.body(name="hande_1").id

        
    def repair_cov(self, C):
        epsilon = 1e-5
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        min_eigenvalue = np.min(eigenvalues)

        if min_eigenvalue < epsilon:
            print("REPAIR COV =========================", flush=True)
            # Clip negative eigenvalues
            clipped = np.where(eigenvalues < epsilon, epsilon, eigenvalues)
            D_prime = np.diag(clipped)
            C_repaired = eigenvectors @ D_prime @ eigenvectors.T
            # C_repaired = (C_repaired + C_repaired.T) / 2
            return C_repaired
        
        # C = (C + C.T) / 2
        return C

        
    def compute_control(self, sim_data, current_pos, current_vel, current_torque):
        """Compute optimal control using CEM/MPC for dual-arm system"""
        
        # Handle covariance matrix numerical stability
        if np.isnan(self.xi_cov).any():
            self.xi_cov = jnp.kron(jnp.eye(self.cem.num_dof), self.cov_coeff_scalar*jnp.identity(self.cem.nvar_single))
        if np.isnan(self.xi_mean).any():
            self.xi_mean = jnp.zeros(self.cem.nvar)

        try:
            np.linalg.cholesky(self.xi_cov)
        except np.linalg.LinAlgError:
            self.xi_cov = jnp.kron(jnp.eye(self.cem.num_dof), self.cov_coeff_scalar*jnp.identity(self.cem.nvar_single))  
        
        # Generate samples
        self.xi_samples, self.key = self.cem.compute_xi_samples(self.key, self.xi_mean, self.xi_cov)

        # self.data = mujoco.MjData(self.model)
        # current_mjx_data = mujoco.mjx.put_data(self.model, self.data)
        self.mjx_model = mujoco.mjx.put_model(self.model)
        current_mjx_data = mujoco.mjx.put_data(self.model, sim_data)

		# ctrl = mjx_data.ctrl.at[jnp.array(self.actuator_ctrl_indices)].set(torque_single)

        current_pos_ = current_mjx_data.qpos.at[self.cem.joint_mask_pos].set(current_pos)
        current_vel_ = current_mjx_data.qvel.at[self.cem.joint_mask_vel].set(current_vel)
        current_torque_ = current_mjx_data.ctrl.at[jnp.array(self.cem.actuator_ctrl_indices)].set(current_torque)



        current_mjx_data = current_mjx_data.replace(qpos = current_pos_, qvel = current_vel_, ctrl=current_torque_)
        # current_mjx_data = current_mjx_data.replace(qpos = current_pos_, qvel = current_vel_)
        # current_mjx_data = jax.jit(mujoco.mjx.forward)(self.mjx_model, current_mjx_data )
        # current_mjx_data = sim_data

        

        # print("current_pos", current_pos.shape)
        # print("current_vel", current_vel.shape)

        # CEM computation
        cost_cem, cost_list_cem, torque_horizon, theta_horizon, \
        self.xi_mean, self.xi_cov, xi_samples, torque_filtered_cem, torque_all, th_all, avg_primal_res, avg_fixed_res, \
        primal_res, fixed_res, idx_min, torso_trace_planned, torso_trace_all  = self.cem.compute_cem(
            current_mjx_data,
            self.xi_mean,
            self.xi_cov,
            current_pos,
            current_vel,
            np.zeros(self.num_dof),  # Zero initial acceleration
            current_torque,
            self.lamda_init,
            self.s_init,
            self.xi_samples,
            self.cost_weights,
        )


        # Get mean velocity command (average middle 90% of trajectory)
        # torque_cem = np.mean(torque_horizon[1:int(0.8*self.num_steps)], axis=0)
        # torque_cem = np.mean(torque_horizon[1:int(0.8*self.num_steps)], axis=0)

        # torque = torque_cem
        torque = torque_horizon
        
        return (torque, cost_cem, cost_list_cem, 
                torque_horizon, 
                theta_horizon, 
                torso_trace_planned,
                torso_trace_all,
                torque_all,
                torque_filtered_cem,
                primal_res,
                fixed_res,
                xi_samples)