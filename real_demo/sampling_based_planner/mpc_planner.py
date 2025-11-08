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
                 inference=False, max_joint_intforce=0.0, max_joint_force=1.0, 
                 max_joint_dforce=5.0, max_joint_ddforce=10.0,
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
        self.inference = inference
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
            max_joint_intforce=max_joint_intforce,
            max_joint_force=max_joint_force,
            max_joint_dforce=max_joint_dforce,
            max_joint_ddforce=max_joint_ddforce
        )
        
        # Initialize CEM variables
        self.cov_scalar_coeff = 0.5
        self.xi_mean_single = jnp.zeros(self.cem.nvar_single)
        self.xi_cov_single = self.cov_scalar_coeff*jnp.identity(self.cem.nvar_single)
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

        
    def compute_control(self, sim_data, current_pos, current_vel, current_force):
        """Compute optimal control using CEM/MPC"""
        
        # Handle covariance matrix numerical stability
        if np.isnan(self.xi_cov).any():
            self.xi_cov = jnp.kron(jnp.eye(self.cem.num_dof), self.cov_scalar_coeff*jnp.identity(self.cem.nvar_single))
        if np.isnan(self.xi_mean).any():
            self.xi_mean = jnp.zeros(self.cem.nvar)

        try:
            np.linalg.cholesky(self.xi_cov)
        except np.linalg.LinAlgError:
            self.xi_cov = jnp.kron(jnp.eye(self.cem.num_dof), self.cov_scalar_coeff*jnp.identity(self.cem.nvar_single))  
        
        # Generate samples
        self.xi_samples, self.key = self.cem.sampling.compute_xi_samples(self.key, self.xi_mean, self.xi_cov)

        # self.xi_samples = jnp.clip(self.xi_samples, a_min=-1.0, a_max=1.0)

        # self.data = mujoco.MjData(self.model)
        # current_mjx_data = mujoco.mjx.put_data(self.model, self.data)
        self.mjx_model = mujoco.mjx.put_model(self.model)
        current_mjx_data = mujoco.mjx.put_data(self.model, sim_data)
        # current_mjx_data = jax.jit(mujoco.mjx.forward)(self.mjx_model, current_mjx_data )
        # current_mjx_data = sim_data

        # CEM computation
        cost_cem, cost_list_cem, force_horizon, joint_pos_horizon, \
        xi_mean, xi_cov, xi_samples, force_filtered_cem, force_all, joint_pos_horizon_all, avg_primal_res, avg_fixed_res, \
        primal_res, fixed_res, idx_min, tip_trace_planned, tip_trace_all  = self.cem.compute_cem(
            current_mjx_data,
            self.xi_mean,
            self.xi_cov,
            current_pos,
            current_vel,
            np.zeros(self.num_dof),  # Zero initial acceleration
            current_force,
            self.lamda_init,
            self.s_init,
            self.xi_samples,
            self.cost_weights,
        )

        
        # Get mean velocity command (average middle 90% of trajectory)
        # force_cem = np.mean(force_horizon[1:20], axis=0)
        # force_cem = np.mean(force_horizon[1:int(0.8*self.num_steps)], axis=0)

        # force = force_cem
        force = force_horizon
        
        return (force, cost_cem, 
                cost_list_cem, 
                force_horizon, 
                joint_pos_horizon,
                joint_pos_horizon_all, 
                tip_trace_planned, 
                tip_trace_all, 
                force_all, 
                force_filtered_cem,
                primal_res,
                fixed_res,
                xi_samples)