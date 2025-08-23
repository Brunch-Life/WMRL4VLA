from typing import Dict, Any, List, Optional, Sequence, Tuple, Union
import json
import numpy as np
import os 
import sapien
import torch
from transforms3d.euler import euler2quat
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from mani_skill.examples.real2sim_3d_assets import ASSET_3D_PATH, REAL2SIM_3D_ASSETS_PATH
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
from mani_skill.envs.utils.randomization.pose import get_single_obj_grid_pose_batch
@register_env("TabletopPickEnv-v1", max_episode_steps=500)
class TabletopPickEnv(BaseEnv):
    """
    This is a simple environment demonstrating pick an object on a table with a robot arm. 
    There are no success/rewards defined, users can using this environment to make panda pick the object.
    """
    SUPPORTED_REWARD_MODES = ["none"]

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        self.object = {"name": [],"actor": [],}
        self.object_name = kwargs.pop("object_name", "tomato")
        self.consecutive_grasp = 0
        self.reconfiguration_num = 0
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # we set contact_offset to a small value as we are not expecting to make any contacts really apart from the brush hitting the canvas too hard.
        # We set solver iterations very low as this environment is not doing a ton of manipulation (the brush is attached to the robot after all)
        return SimConfig(
            sim_freq=500,
            control_freq=5,
            scene_config=SceneConfig(
                contact_offset=0.01,
                solver_position_iterations=4,
                solver_velocity_iterations=0,
            ),
        )

    @property
    def _default_sensor_configs(self):
        # agent was set by self.table_scene.initialize()
        robot_pose = sapien.Pose(p=[-0.615, 0, 0], q=[1,0,0,0])
        eye = torch.tensor([0.76357918+robot_pose.p[0], -0.0395012+robot_pose.p[1], 0.68071344+robot_pose.p[2]])
        rotation = torch.tensor([
                    [-0.53301526, -0.05021884, -0.844614,],
                    [0.01688062, -0.99866954, 0.04872569,],
                    [-0.84593722, 0.01171393, 0.53315383,],
                ])
        pose = Pose.create_from_pq(p=eye, q=matrix_to_quaternion(rotation))
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=640,
                height=480,
                fov=np.deg2rad(44), # vertical fov for realsense d435
                near=0.01,
                far=100,
                # mount=sapien_utils.get_obj_by_name(self.agent.robot.get_links(), "panda_link0"),
            ),
            CameraConfig(
                "3rd_view_camera",
                pose=pose,
                width=640,
                height=480,
                fov=np.deg2rad(44), # vertical fov for realsense d435
                near=0.01,
                far=100,
                # mount=sapien_utils.get_obj_by_name(self.agent.robot.get_links(), "panda_link0"),
            )
        ]

    @property
    def _default_human_render_camera_configs(self): # what we use to render the scene
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=640, # 1280
            height=480, # 960
            fov=1.2,
            near=0.01,
            far=100,
        )

    def get_language_instruction(self):
        object_name = self.object["name"][0]
        return [f"Pick the {object_name} up."] * self.num_envs

    def get_scene_description(self):
        return [f"The scene is a simulated workspace designed for robotics tasks. It centers around a wooden table surface situated within a plain, neutral-colored room. A robotic arm is positioned above the table, ready to interact with the environment."] * self.num_envs

    # not useful, for the pose being changed in _initial_episode
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _builder_object_helper(self, obj_path_root_path: str,obj_path: str,quat, scale = 1.0):
        assests_scale_data_path = REAL2SIM_3D_ASSETS_PATH+"/assets_scale.json"
        with open(assests_scale_data_path, "r", encoding="utf-8") as f:
            assests_scale_data = json.load(f)
        object_name = obj_path.split('.')[0]
        if object_name in assests_scale_data.keys():
            scale = np.array(assests_scale_data[object_name]["scale"])
            quat = np.array(assests_scale_data[object_name]["quat"])
        else:
            scale = [scale] * 3
        obj_abs_path = os.path.join(obj_path_root_path, obj_path)   
        builder = self.scene.create_actor_builder()
        builder.set_mass_and_inertia(
            mass=0.1,
            cmass_local_pose=sapien.Pose([0,0,0],q=quat),
            inertia=[0,0,0], 
        )
        builder.add_multiple_convex_collisions_from_file(
            filename=obj_abs_path,
            scale=scale,
            pose=sapien.Pose(p=[0, 0, 0],q=quat),
            decomposition="coacd"
        )
        builder.add_visual_from_file(
            filename=obj_abs_path,
            scale=scale,
            pose=sapien.Pose(p=[0, 0, 0],q=quat),
        )
        xyz, quat = get_single_obj_grid_pose_batch(np.array([-0.2, 0.0]), 0.1, 0.25, 0, 0.01,self.num_envs)
        initial_pose = [sapien.Pose(p=xyz[i], q=quat[i]) for i in range(self.num_envs)] # sapien should use this, but Pose in Maniskill can use batch 
        builder.set_initial_pose(initial_pose)
        return builder.build_dynamic(name=object_name)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build(is_table_green = False)
        self.object = {"name": [],"actor": [],}
        obj_path_root_path = ASSET_3D_PATH
        obj_path_list = os.listdir(obj_path_root_path)
        obj_path_list = list(filter(lambda x: not x.endswith('.ply'), obj_path_list)) # pop the ply file path
        q_x_90 = euler2quat(np.pi / 2, 0, 0).astype(np.float32)
        # get and create object
        # obj_idx = self.reconfiguration_num % len(obj_path_list)
        obj_idx = [i for i, v in enumerate(obj_path_list) if v == self.object_name+".glb"][0]
        self.object["name"].append(obj_path_list[obj_idx].split('.')[0])
        actor = self._builder_object_helper(obj_path_root_path, obj_path_list[obj_idx], q_x_90, 1)
        self.object["actor"].append(actor)
        print("created object: ", self.object["name"], f"obj_idx = {obj_idx}")
        self.reconfiguration_num += 1 # now you should make sure every planning is perfect.
        if self.reconfiguration_num == len(obj_path_list):
            pass

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.set_initial_qpos(env_idx)
            episode_idx = None
            if "episode_id" in options:
                if isinstance(options["episode_id"], int):
                    episode_idx = options["episode_id"]
            for i, actor in enumerate(self.object["actor"]):
                xyz, quat = get_single_obj_grid_pose_batch(np.array([-0.2, 0.0]), 0.1, 0.25, 0, 0.01,self.num_envs, episode_idx)
                actor.set_pose(Pose.create_from_pq(p=xyz, q=quat))
            # if self.gpu_sim_enabled:
            #     self.scene._gpu_apply_all()
            # self._settle(0.5)
            # if self.gpu_sim_enabled:
            #     self.scene._gpu_fetch_all()
            # # Some objects need longer time to settle
            # lin_vel, ang_vel = 0.0, 0.0
            # for i, actor in enumerate(self.object["actor"]):
            #     lin_vel += torch.linalg.norm(actor.linear_velocity)
            #     ang_vel += torch.linalg.norm(actor.angular_velocity)
            # if lin_vel > 1e-3 or ang_vel > 1e-2:
            #     if self.gpu_sim_enabled:
            #         self.scene._gpu_apply_all()
            #     self._settle(6)
            #     if self.gpu_sim_enabled:
            #         self.scene._gpu_fetch_all()

    def _settle(self, t: int = 0.5):
        """run the simulation for some steps to help settle the objects"""
        sim_steps = int(self.sim_freq * t / self.control_freq)
        for _ in range(sim_steps):
            self.scene.step()

    def set_initial_qpos(self, env_idx):
        robot_init_qpos_noise = 0
        b = len(env_idx)
        # fmt: off
        qpos = np.array(
            [-0.1788885, -0.5299233, 0.21601543, -2.9509537, 0.16559684, 2.4244094, 0.6683393, 0.04, 0.04],
        )
        # fmt: on
        if self._enhanced_determinism:
            qpos = (
                self._batched_episode_rng[env_idx].normal(
                    0, robot_init_qpos_noise, len(qpos)
                )
                + qpos
            )
        else:
            qpos = (
                self._episode_rng.normal(
                    0, robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
        qpos[:, -2:] = 0.04
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    def _get_obs_extra(self, info: Dict):
        """Get task-relevant extra observations. Usually defined on a task by task basis"""
        return dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
    
    def evaluate(self,) -> dict:
        """
        Evaluate whether the environment is currently in a success state by returning a dictionary with a "success" key or
        a failure state via a "fail" key
        """
        is_grasped = self.agent.is_grasping(self.object["actor"][0])
        self.consecutive_grasp += is_grasped
        self.consecutive_grasp[is_grasped == 0] = 0
        consecutive_grasp = self.consecutive_grasp >= 5
        # self.scene.get_pairwise_contact_forces()
        # maybe can add a z_offset to ensure the pick movement

        return dict(is_grasped=is_grasped, success=consecutive_grasp,  consecutive_grasp=consecutive_grasp)


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.object["actor"][0].pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        return reward
