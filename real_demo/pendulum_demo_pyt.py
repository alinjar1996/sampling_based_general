import os
import time
import numpy as np

import mujoco
from mujoco import viewer


from sampling_based_planner.mpc_planner import run_cem_planner
from sampling_based_planner.quat_math import *

np.set_printoptions(precision=4, suppress=True)


class Planner:

    def __init__(
        self,
        record_data=False,
        idx=0,
        num_batch=500,
        num_steps=15,
        maxiter_cem=2,
        maxiter_projection=5,
        num_elite=0.05,
        timestep=0.1,
    ):

        self.record_data_ = record_data
        self.idx = str(idx).zfill(2)

        self.num_dof = 1
        self.init_joint_position = np.array([0.0])
        self.num_steps = num_steps
        self.timestep = timestep

        cost_weights = {
            "theta": 1.0,
            "thetadot": 0.01,
            "control": 0.001,
        }

        self.torque = np.zeros(self.num_dof)
        self.torque_array = np.zeros(self.num_dof * num_steps)

        # MuJoCo model
        # model_path = os.path.join(
        #     get_package_share_directory("real_demo"),
        #     "pendulum_mjx",
        #     "scene.xml",
        # )

        model_path = "real_demo/pendulum_mjx/scene.xml"

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.timestep
        self.data = mujoco.MjData(self.model)

        mujoco.mj_forward(self.model, self.data)

        # Joint masks
        joint_names_pos = []
        joint_names_vel = []
        joint_names_ctrl = []

        for i in range(self.model.njnt):

            joint_type = self.model.jnt_type[i]

            n_pos = (
                7
                if joint_type == mujoco.mjtJoint.mjJNT_FREE
                else 4
                if joint_type == mujoco.mjtJoint.mjJNT_BALL
                else 1
            )

            n_vel = (
                6
                if joint_type == mujoco.mjtJoint.mjJNT_FREE
                else 3
                if joint_type == mujoco.mjtJoint.mjJNT_BALL
                else 1
            )

            n_ctrl = n_vel

            for _ in range(n_pos):
                joint_names_pos.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                )

            for _ in range(n_vel):
                joint_names_vel.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                )

            for _ in range(n_ctrl):
                joint_names_ctrl.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                )

        robot_joints = np.array(["pendulum_joint"])

        self.joint_mask_pos = np.isin(joint_names_pos, robot_joints)
        self.joint_mask_vel = np.isin(joint_names_vel, robot_joints)
        self.joint_mask_ctrl = np.isin(joint_names_ctrl, robot_joints)

        self.actuator_joint_ids = self.model.actuator_trnid[:, 0]

        self.actuator_ctrl_indices = [
            i
            for i, j in enumerate(self.actuator_joint_ids)
            if self.joint_mask_ctrl[j]
        ]

        self.data.qpos[self.joint_mask_pos] = self.init_joint_position

        # Planner
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
            cost_weights=cost_weights,
        )

        # Viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.cam.distance = 5.0
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -30.0

        self.traj_time_start = time.time()

    def render_trace(self, viewer_, tip_trace_positions):

        viewer_.user_scn.ngeom = 0

        for i, pos in enumerate(tip_trace_positions):

            geom_id = viewer_.user_scn.ngeom
            viewer_.user_scn.ngeom += 1

            size = np.array([0.02, 0.02, 0.02])
            pos = np.array(pos).reshape(3)
            mat = np.eye(3).flatten()

            t = i / (len(tip_trace_positions) - 1 + 1e-9)

            start_color = np.array([1, 0, 0, 1])
            end_color = np.array([0, 1, 0, 1])

            rgba = (1 - t) * start_color + t * end_color

            mujoco.mjv_initGeom(
                viewer_.user_scn.geoms[geom_id],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                pos,
                mat,
                rgba,
            )

    def control_loop(self):

        start_time = time.time()

        current_pos = self.data.qpos[self.joint_mask_pos]
        current_vel = self.data.qvel[self.joint_mask_vel]

        self.torque = np.mean(self.torque_array[1:5], axis=0)

        (self.torque_array,
            cost_cem,
            cost_list_cem,
            torque_horizon,
            theta_horizon,
            tip_trace_planned,
            tip_trace_all,
            torque_samples,
            torque_filtered,
            primal_res,
            fixed_res,
            xi_samples,
        ) = self.planner.compute_control(
            self.data,
            current_pos,
            current_vel,
            self.torque)

        cost_theta, cost_thetadot, cost_control = cost_list_cem[-1]

        self.data.ctrl[self.actuator_ctrl_indices] = self.torque

        mujoco.mj_step(self.model, self.data)

        self.render_trace(self.viewer, tip_trace_planned[:, :3])

        self.viewer.sync()

        print(
            f"\n| Total time: {time.time()-self.traj_time_start:.1f}s"
            f"\n| Step time: {(time.time()-start_time)*1000:.1f}ms"
            f"\n| Cost: {cost_cem[-1]:.2f}"
            f"\n| Theta: {cost_theta:.2f}"
            f"\n| ThetaDot: {cost_thetadot:.2f}"
            f"\n| Control: {cost_control:.2f}"
        )

        step_sleep = self.model.opt.timestep - (time.time() - start_time)

        if step_sleep > 0:
            time.sleep(step_sleep)

    def run(self):

        while self.viewer.is_running():
            self.control_loop()


def main():

    planner = Planner(
        record_data=False,
        num_batch=500,
        num_steps=15,
        maxiter_cem=1,
        maxiter_projection=5,
        num_elite=0.05,
        timestep=0.1,
    )

    planner.run()


if __name__ == "__main__":
    main()