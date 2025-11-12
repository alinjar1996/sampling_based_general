from sampling_based_planner.mjx_planner import cem_planner
from sampling_based_planner.quat_math import quaternion_distance, quaternion_multiply, rotation_quaternion
from ik_based_planner.ik_solver import InverseKinematicsSolver
from MLP_projection.mlp_biped_torque import MLP, MLPProjectionFilter

import mujoco
from mujoco import viewer
import jax.numpy as jnp
import jax

import numpy as np
import torch 
import contextlib
from io import StringIO

import os
from ament_index_python.packages import get_package_share_directory





class run_cem_planner:
    def __init__(self, model, data, num_dof=None, num_batch=None, num_steps=None, 
                 maxiter_cem=None, maxiter_projection=None, num_elite=None, timestep=None,
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

        self.max_joint_torque = max_joint_torque
        self.max_joint_dtorque = max_joint_dtorque
        self.max_joint_ddtorque = max_joint_ddtorque

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
        self.cov_coeff_scalar = 1.0
        self.xi_mean_single = jnp.zeros(self.cem.nvar_single)
        self.xi_cov_single = self.cov_coeff_scalar*jnp.identity(self.cem.nvar_single)
        self.xi_mean = jnp.tile(self.xi_mean_single, self.cem.num_dof)
        self.xi_cov = jnp.kron(jnp.eye(self.cem.num_dof), self.xi_cov_single)
        self.lamda_init = jnp.zeros((num_batch, self.cem.nvar))
        self.s_init = jnp.zeros((num_batch, self.cem.num_total_constraints))
        self.key = jax.random.PRNGKey(0)
        
        self.inference = False


        # Get absolute path to the package share folder
        package_share = get_package_share_directory('real_demo')

        # Build path to your weights
        self.weight_path = os.path.join(
            package_share,
            'mlp_proj_training_weights',
            f'mlp_biped_2000_100_1_{self.cem.num_batch}.pth'
        )

        # print(f"[INFO] Loading MLP weights from: {self.weight_path}")

        # self.inp = jnp.hstack([self.cem.compute_xi_samples])
        # Initialize MLP if inference is enabled
        if self.inference:
            self.mlp_model = self._load_mlp_projection_model((num_steps+1)*num_dof, maxiter_projection, self.weight_path)
        
    
    def _load_mlp_projection_model(self, num_feature, maxiter_projection, weight_path):
        """Load the MLP projection model for inference"""
        enc_inp_dim = num_feature
        mlp_inp_dim = enc_inp_dim
        hidden_dim = 1024
        mlp_out_dim = 2 * self.cem.nvar

        mlp = MLP(mlp_inp_dim, hidden_dim, mlp_out_dim)

        # model = MLPProjectionFilter(mlp, P, Pdot, Pddot, num, num_batch_train, inp_mean, inp_std, t_fin).to(device)

        with contextlib.redirect_stdout(StringIO()):
            model = MLPProjectionFilter(
                mlp=mlp,
                P = torch.tensor(np.array(self.cem.P), dtype=torch.float32),
                Pdot = torch.tensor(np.array(self.cem.Pdot), dtype=torch.float32),
                Pddot = torch.tensor(np.array(self.cem.Pddot), dtype=torch.float32),
                num = self.num_steps,
                num_batch = self.cem.num_batch,
                max_joint_torque = self.max_joint_torque,
                max_joint_dtorque = self.max_joint_dtorque,
                max_joint_ddtorque = self.max_joint_ddtorque,
                t_fin = self.timestep*self.num_steps,
                maxiter_projection=self.cem.maxiter_projection,
            ).to(self.device)

            
    
            model.load_state_dict(torch.load(weight_path, weights_only=True))
            model.eval()
        
        return model

        
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
        self.xi_samples, self.key = self.cem.sampling.compute_xi_samples(self.key, self.xi_mean, self.xi_cov)

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

        if self.inference:
            xi_projected_nn_output = []
            lamda_init_nn_output = []
            s_init_nn_output = []

            init_control = current_torque
            # init_control_repeated = np.tile(init_control, (self.cem.num_batch,1))
            init_control_repeated = np.broadcast_to(init_control, (self.cem.num_batch, init_control.size))


            inp = np.hstack((init_control_repeated, self.xi_samples))
            inp_torch = torch.tensor(inp).float().to(self.device)

            inp_mean = inp.mean()
            inp_std = inp.std()
            inp_norm = (inp - inp_mean) / inp_std
            inp_norm_torch = torch.tensor(inp_norm).float().to(self.device)

            neural_output_batch = self.mlp_model.mlp(inp_norm_torch)

            #xi_samples_nn_output = 
            xi_samples_nn_output = neural_output_batch[:, 0:self.cem.nvar].to(self.device)  
            lamda_init_nn_output = neural_output_batch[:, self.cem.nvar:2*self.cem.nvar].to(self.device)  
            s_init_nn_output = neural_output_batch[:, 2*self.cem.nvar:2*self.cem.nvar+self.cem.num_total_constraints].to(self.device)

            self.xi_samples = np.array(xi_samples_nn_output.cpu().detach().numpy())
            self.lamda_init = np.array(lamda_init_nn_output.cpu().detach().numpy())
            self.s_init = np.array(s_init_nn_output.cpu().detach().numpy())

            
            


        # CEM computation
        cost_cem, cost_list_cem, torque_horizon, theta_horizon, \
        xi_mean, xi_cov, xi_samples, xi_filtered, torque_all, th_all, avg_primal_res, avg_fixed_res, \
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
        
        # # If the below lines are uncommented, mean and covariance of sampiling variables get updated with time, otherwise it does not.
        # self.xi_mean = xi_mean
        # self.xi_cov = xi_cov


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
                xi_filtered,
                primal_res,
                fixed_res,
                xi_samples)