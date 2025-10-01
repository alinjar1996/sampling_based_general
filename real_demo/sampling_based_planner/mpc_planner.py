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
                 maxiter_cem=1, maxiter_projection=5, num_elite=0.05, timestep=0.05,
                 position_threshold=0.06, rotation_threshold=0.1,
                 ik_pos_thresh=0.08, ik_rot_thresh=0.1, 
                 collision_free_ik_dt=2.0, inference=False, rnn=None,
                 max_joint_pos=180.0*np.pi/180.0, max_joint_vel=1.0, 
                 max_joint_acc=2.0, max_joint_jerk=4.0,
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
        # self.position_threshold = position_threshold
        # self.rotation_threshold = rotation_threshold
        # self.ik_pos_thresh = ik_pos_thresh if ik_pos_thresh else 1.1 * position_threshold
        # self.ik_rot_thresh = ik_rot_thresh if ik_rot_thresh else 1.1 * rotation_threshold
        # self.collision_free_ik_dt = collision_free_ik_dt
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
            max_joint_pos=max_joint_pos,
            max_joint_vel=max_joint_vel,
            max_joint_acc=max_joint_acc,
            max_joint_jerk=max_joint_jerk,

        )
        
        # Initialize CEM variables
        self.xi_mean_single = jnp.zeros(self.cem.nvar_single)
        self.xi_cov_single = 10*jnp.identity(self.cem.nvar_single)
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

        self.torso = model.site(name="torso_site").id
        
        

    # def update_targets(self, target_idx=0, target_pos=None, target_rot=None):
    #     """Update target positions and rotations for both arms"""
    #     if target_idx==0:
    #         self.target_pos_0 = target_pos
    #         self.target_rot_0 = target_rot
    #         self.target_0 = np.concatenate([self.target_pos_0, self.target_rot_0])
    #     elif target_idx==1:
    #         self.target_pos_1 = target_pos
    #         self.target_rot_1 = target_rot
    #         self.target_1 = np.concatenate([self.target_pos_1, self.target_rot_1])
        
    # def update_obstacle(self, obstacle_pos, obstacle_rot):
    #     """Update obstacle position and rotation"""
    #     self.obstacle_pos = obstacle_pos
    #     self.obstacle_rot = obstacle_rot
        
    def compute_control(self, current_pos, current_vel):
        """Compute optimal control using CEM/MPC for dual-arm system"""
        
        # Handle covariance matrix numerical stability
        if np.isnan(self.xi_cov).any():
            self.xi_cov = jnp.kron(jnp.eye(self.cem.num_dof), 10*jnp.identity(self.cem.nvar_single))
        if np.isnan(self.xi_mean).any():
            self.xi_mean = jnp.zeros(self.cem.nvar)

        try:
            np.linalg.cholesky(self.xi_cov)
        except np.linalg.LinAlgError:
            self.xi_cov = jnp.kron(jnp.eye(self.cem.num_dof), 10*jnp.identity(self.cem.nvar_single))  
        
        # Generate samples
        self.xi_samples, self.key = self.cem.compute_xi_samples(self.key, self.xi_mean, self.xi_cov)


        # CEM computation
        cost, best_cost_list, thetadot_horizon, theta_horizon, \
        self.xi_mean, self.xi_cov, thd_all, th_all, avg_primal_res, avg_fixed_res, \
        primal_res, fixed_res, idx_min, torso_trace_planned, torso_trace  = self.cem.compute_cem(
            self.xi_mean,
            self.xi_cov,
            current_pos,
            current_vel,
            np.zeros(self.num_dof),  # Zero initial acceleration
            self.lamda_init,
            self.s_init,
            self.xi_samples,
            self.cost_weights,
        )

        # Get mean velocity command (average middle 90% of trajectory)
        thetadot_cem = np.mean(thetadot_horizon[1:6], axis=0)

        thetadot_0 = thetadot_cem[:6]
        thetadot_1 = thetadot_cem[6:]
        

        # Combine control commands
        thetadot = np.concatenate((thetadot_0, thetadot_1))
        
        return thetadot, cost, best_cost_list, thetadot_horizon, theta_horizon, torso_trace_planned