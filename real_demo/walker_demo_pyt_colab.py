import os
os.environ["MUJOCO_GL"] = "egl"

import time
import numpy as np
import mujoco
import imageio
import argparse

from sampling_based_planner.mpc_planner import run_cem_planner
from sampling_based_planner.quat_math import *

np.set_printoptions(precision=4, suppress=True)


class Planner:

    def __init__(
        self,
        num_batch=500,
        num_steps=15,
        maxiter_cem=1,
        maxiter_projection=5,
        num_elite=0.05,
        timestep=0.1,
    ):

        self.num_dof = 6
        self.init_joint_position = np.zeros(6)
        self.timestep = timestep

        cost_weights = {
            "height": 20.0,
            "orientation": 15.0,
            "velocity": 15.0,
            "control": 0.1,
        }

        self.torque = np.zeros(self.num_dof)
        self.torque_array = np.zeros((num_steps, self.num_dof))

        model_path = "real_demo/walker_mjx/scene.xml"

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)

        mujoco.mj_forward(self.model, self.data)

        # renderer
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.frames = []

        # Sensors
        self.torso_position_sensor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position"
        )

        self.torso_velocity_sensor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel"
        )

        self.torso_zaxis_sensor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_zaxis"
        )

        print(
            "Sensors:",
            self.torso_position_sensor,
            self.torso_velocity_sensor,
            self.torso_zaxis_sensor,
        )

        robot_joints = np.array(
            [
                "right_hip",
                "right_knee",
                "right_ankle",
                "left_hip",
                "left_knee",
                "left_ankle",
            ]
        )

        joint_names_pos = []
        joint_names_vel = []
        joint_names_ctrl = []

        for i in range(self.model.njnt):

            joint_type = self.model.jnt_type[i]

            n_pos = (
                7 if joint_type == mujoco.mjtJoint.mjJNT_FREE
                else 4 if joint_type == mujoco.mjtJoint.mjJNT_BALL
                else 1
            )

            n_vel = (
                6 if joint_type == mujoco.mjtJoint.mjJNT_FREE
                else 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL
                else 1
            )

            for _ in range(n_pos):
                joint_names_pos.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                )

            for _ in range(n_vel):
                joint_names_vel.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                )

            for _ in range(n_vel):
                joint_names_ctrl.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                )

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

        self.planner = run_cem_planner(
            model=self.model,
            data=self.data,
            num_dof=self.num_dof,
            num_batch=num_batch,
            num_steps=num_steps,
            maxiter_cem=maxiter_cem,
            maxiter_projection=maxiter_projection,
            num_elite=num_elite,
            timestep=timestep,
            cost_weights=cost_weights,
            inference_jax=True,
        )

        self.traj_time_start = time.time()

    # sensor helper
    def get_sensor_value(self, sensor_id):

        sensor_data = self.data.sensordata
        sensor_adr = self.model.sensor_adr[sensor_id]
        sensor_dim = self.model.sensor_dim[sensor_id]

        if sensor_dim == 1:
            return sensor_data[sensor_adr]

        return sensor_data[sensor_adr : sensor_adr + sensor_dim]

    # trace rendering
    def render_trace(self, torso_trace_positions):

        scene = self.renderer.scene
        base = scene.ngeom

        for i, pos in enumerate(torso_trace_positions):

            geom_id = base + i
            if geom_id >= scene.maxgeom:
                break

            scene.ngeom += 1

            size = np.array([0.02, 0.02, 0.02])
            pos = np.array(pos).reshape(3)
            mat = np.eye(3).flatten()

            t = i / (len(torso_trace_positions) - 1 + 1e-9)

            start_color = np.array([1, 0, 0, 1])
            end_color = np.array([0, 1, 0, 1])
            rgba = (1 - t) * start_color + t * end_color

            mujoco.mjv_initGeom(
                scene.geoms[geom_id],
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

        (
            self.torque_array,
            cost_cem,
            cost_list_cem,
            torque_horizon,
            theta_horizon,
            torso_trace_planned,
            *_,
        ) = self.planner.compute_control(
            self.data,
            current_pos,
            current_vel,
            self.torque,
        )

        cost_height, cost_orientation, cost_velocity, cost_control = cost_list_cem[-1]

        self.data.ctrl[self.actuator_ctrl_indices] = self.torque

        mujoco.mj_step(self.model, self.data)

        torso_pos = self.get_sensor_value(self.torso_position_sensor)

        print("\nTorso Position:", torso_pos)

        self.renderer.update_scene(self.data)

        self.render_trace(torso_trace_planned[:, :3])

        frame = self.renderer.render()

        self.frames.append(frame)

        print(
            f"\n| Total time: {time.time()-self.traj_time_start:.1f}s"
            f"\n| Step time: {(time.time()-start_time)*1000:.1f}ms"
            f"\n| Total Cost (Across Horizon): {cost_cem[-1]:.2f}"
            f"\n| Cost Height (Across Horizon): {cost_height:.2f}"
            f"\n| Cost Orientation (Across Horizon): {cost_orientation:.2f}"
            f"\n| Velocity (Across Horizon): {cost_velocity:.2f}"
            f"\n| Control (Across Horizon): {cost_control:.2f}",
            flush=True,
        )

        step_sleep = self.model.opt.timestep - (time.time() - start_time)

        if step_sleep > 0:
            time.sleep(step_sleep)

    def run(self, steps=250):

        for i in range(steps):

            print(f"\nStep {i}")

            self.control_loop()

        imageio.mimsave("walker_video.mp4", self.frames, fps=10)

        print("\nVideo saved: walker_video.mp4")
    
def main():

    parser = argparse.ArgumentParser(description="Pendulum CEM MPC Planner")
    parser.add_argument("--num_batch", type=int, default=25)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--maxiter_cem", type=int, default=5)
    parser.add_argument("--maxiter_projection", type=int, default=1)
    parser.add_argument("--num_elite", type=float, default=0.05)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    planner = Planner(
        num_batch=args.num_batch,
        num_steps=args.num_steps,
        maxiter_cem=args.maxiter_cem,
        maxiter_projection=args.maxiter_projection,
        num_elite=args.num_elite,
        timestep=args.timestep,
    )

    planner.run(steps=args.steps)



if __name__ == "__main__":
    main()