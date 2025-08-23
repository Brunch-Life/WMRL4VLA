from typing import Dict, Any, List, Optional, Sequence, Tuple, Union
import json
import numpy as np
import os 
import sapien
import torch
from transforms3d.euler import euler2quat
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.examples.real2sim_3d_assets import ASSET_3D_PATH, REAL2SIM_3D_ASSETS_PATH, CONTAINER_3D_PATH
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam, Panda
from mani_skill.envs.utils.randomization.pose import get_objs_random_pose_batch, get_single_obj_grid_pose_batch, get_objs_deterministic_pose_batch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion

@register_env("TabletopPickPlaceEnv-v1", max_episode_steps=500)
class TabletopPickPlaceEnv(BaseEnv):
    """
    This is a simple environment demonstrating pick an object on a table with a robot arm. 
    There are no success/rewards defined, users can using this environment to make panda pick the object.
    """
    SUPPORTED_REWARD_MODES = ["none"]

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam",]
    agent: Union[Panda, PandaWristCam,]

    def __init__(self, *args, robot_uids="panda", **kwargs):
        object_name = kwargs.pop("object_name", None)
        self.object_name = "tomato" if object_name is None else object_name
        container_name = kwargs.pop("container_name", None)
        self.container_name = "plate" if container_name is None else container_name

        self.is_table_green = kwargs.pop("is_table_green", False)
        self.object = {"name": [],"actor": [],}
        self.consecutive_grasp = 0
        self.reconfiguration_num = 0

        if robot_uids == "panda" or robot_uids == "panda_wristcam":
            self.xy_center = np.array([[-0.17, 0.0],[-0.17, 0.0]])
        else:
            self.xy_center = np.array([[-0.17, 0.0],[-0.17, 0.0]])
        self.half_edge_length_x = np.array([0.1, 0.1])
        self.half_edge_length_y = np.array([0.2, 0.2])

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
        if self.robot_uids == "panda_wristcam" or self.robot_uids == "panda":
            camera_config_base = CameraConfig(
                "base_camera",
                pose=pose,
                width=640,
                height=480,
                fov=np.deg2rad(44), # vertical fov for realsense d435
                near=0.01,
                far=100,
            )    
            camera_config = CameraConfig(
                    "3rd_view_camera",
                    pose=pose,
                    width=640,
                    height=480,
                    fov=np.deg2rad(44), # vertical fov for realsense d435
                    near=0.01,
                    far=100,
                )
        return [camera_config, camera_config_base]

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

    def get_language_instruction(self, is_place = True):
        object_name = self.object["name"][0].replace("_", " ")
        if is_place:
            container_name = self.object["name"][1].replace("_", " ")
            return [f"Put {object_name} on {container_name}."] * self.num_envs
        return [f"Pick {object_name} up."] * self.num_envs

    def get_scene_description(self):
        return [f"The scene is a simulated workspace designed for robotics tasks. It centers around a wooden table surface situated within a plain, neutral-colored room. A robotic arm is positioned above the table, ready to interact with the environment."] * self.num_envs

    # not useful, for the pose being changed in _initial_episode
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _build_container(self):
        container_root_path = CONTAINER_3D_PATH
        container_path_list = os.listdir(container_root_path)
        container_path_list = list(filter(lambda x: not x.endswith('.ply'), container_path_list)) # pop the ply file path
        q_x_90 = euler2quat(np.pi / 2, 0, 0).astype(np.float32)
        name = self.object["name"].append(container_path_list[0].split('/')[0])
        container_actor = self._builder_object_helper(container_root_path, container_path_list[0], q_x_90, 1)
        return name, container_actor

    def get_true_random_pose_batch(self,options=None):            
        b = self.num_envs
        source_extents = self.object["actor"][0].get_first_collision_mesh(to_world_frame=False).bounding_box_oriented.extents.copy()
        target_extents = self.object["actor"][1].get_first_collision_mesh(to_world_frame=False).bounding_box_oriented.extents.copy()
        extents_x = [source_extents[0], target_extents[0]]
        extents_y = [source_extents[1], target_extents[1]]

        z_value = np.array([0,0])
        quats = [None, None]

        if options is not None and "episode_id" in options:
            if isinstance(options["episode_id"], int):
                options["episode_id"] = torch.tensor([options["episode_id"]])
            assert len(options["episode_id"]) == b, "batch size not match"
            source_obj_xyz, source_obj_quat, target_obj_xyz, target_obj_quat = get_objs_deterministic_pose_batch(
                self.xy_center, self.half_edge_length_x, self.half_edge_length_y,
                z_value, extents_x, extents_y, quats, b, options["episode_id"], grid_size=500, threshold_scale=0.7, use_continuous_positions=True
            )
        else:
            # if options is None, we use the random pose
            source_obj_xyz, source_obj_quat, target_obj_xyz, target_obj_quat = get_objs_random_pose_batch(
                    self.xy_center, self.half_edge_length_x, self.half_edge_length_y,
                    z_value, extents_x, extents_y, quats, b, 0.7,
                )
        return source_obj_xyz, source_obj_quat, target_obj_xyz, target_obj_quat

    def _builder_object_helper(self, obj_path_root_path: str,obj_path: str, quat, scale = 1.0):
        assests_scale_data_path = REAL2SIM_3D_ASSETS_PATH+"/assets_scale.json"
        with open(assests_scale_data_path, "r", encoding="utf-8") as f:
            assests_scale_data = json.load(f)
        object_name = obj_path.split('.')[0]
        if object_name in assests_scale_data.keys():
            if "scale" in assests_scale_data[object_name]:
                scale = np.array(assests_scale_data[object_name]["scale"])
            if "quat" in assests_scale_data[object_name]:
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
        xyz, quat = get_single_obj_grid_pose_batch(np.array([-0.2, 0.0]), 0.1, 0.2, 0, 0.01,self.num_envs)
        initial_pose = [sapien.Pose(p=xyz[i], q=quat[i]) for i in range(self.num_envs)] # sapien should use this, but Pose in Maniskill can use batch 
        builder.set_initial_pose(initial_pose)
        return builder.build_dynamic(name=object_name)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build(is_table_green=self.is_table_green)
        self.object = {"name": [],"actor": [],}

        # container
        container_path_root_path = CONTAINER_3D_PATH
        container_path_list = os.listdir(container_path_root_path)
        container_path_list = list(filter(lambda x: not x.endswith('.ply'), container_path_list)) # pop the ply file path

        # object
        obj_path_root_path = ASSET_3D_PATH
        obj_path_list = os.listdir(obj_path_root_path)
        obj_path_list = list(filter(lambda x: not x.endswith('.ply'), obj_path_list)) # pop the ply file path
        q_x_90 = euler2quat(np.pi / 2, 0, 0).astype(np.float32)
        # # get and create object
        # obj_idx = self.reconfiguration_num % len(obj_path_list)

        """
            source_obj_name: "blueberry", "garlic", "golf_ball", "green_bell_pepper",
                "lemon", "magnifying_glass_with_victorian_wooden_handle", "nonstop",
                "peach", "pen", "red_apple", "tomato", "yellow_mongo"

            container_name: "plate", "bowl"
        """

        source_obj_name = self.object_name
        container_name = self.container_name

        source_idx = [i for i, v in enumerate(obj_path_list) if v == source_obj_name+".glb"][0]
        container_idx = [i for i, v in enumerate(container_path_list) if v == container_name+".glb"][0]

        # source_object
        self.object["name"].append(obj_path_list[source_idx].split('.')[0])
        actor = self._builder_object_helper(obj_path_root_path, obj_path_list[source_idx], q_x_90, 1)
        self.object["actor"].append(actor)

        # container
        self.object["name"].append(container_path_list[container_idx].split('.')[0])
        actor = self._builder_object_helper(container_path_root_path, container_path_list[container_idx], q_x_90, 1)
        self.object["actor"].append(actor)

        print("created object: ", self.object["name"], f"source_idx = {source_idx}, container_idx = {container_idx} ")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            assert b==self.num_envs, "envs_num not equal to batch_size"
            self.table_scene.initialize(env_idx)
            self.set_initial_qpos(env_idx)

            source_obj_xyz, source_obj_quat, target_obj_xyz, target_obj_quat = self.get_true_random_pose_batch(options=options)    
            self.object["actor"][0].set_pose(Pose.create_from_pq(p=source_obj_xyz, q=source_obj_quat))
            self.object["actor"][1].set_pose(Pose.create_from_pq(p=target_obj_xyz, q=target_obj_quat))

            # if "episode_id" in options:
            #     if isinstance(options["episode_id"][0].item(), int):
            #         episode_idx = options["episode_id"]
            #         source_obj_xyz, source_obj_quat, target_obj_xyz, target_obj_quat = self.get_true_random_pose_batch()
            #         self.object["actor"][0].set_pose(Pose.create_from_pq(p=source_obj_xyz, q=source_obj_quat))
            #         self.object["actor"][1].set_pose(Pose.create_from_pq(p=target_obj_xyz, q=target_obj_quat))
                    # for i, actor in enumerate(self.object["actor"]):
                    #     xyz, quat = self.get_random_pose(episode_idx)
                    #     actor.set_pose(Pose.create_from_pq(p=xyz, q=quat))

                    # # The code below will cause error
                    # if self.gpu_sim_enabled:
                    #     self.scene._gpu_apply_all()
                    # self._settle(0.5)
                    # if self.gpu_sim_enabled:
                    #     self.scene._gpu_fetch_all()

            # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
            """source object bbox size (3, )"""
            self.episode_source_obj_bbox_world = torch.from_numpy(self.object["actor"][0].get_first_collision_mesh(to_world_frame=True).bounding_box_oriented.extents.copy())
            """target object bbox size (3, )"""
            self.episode_target_obj_bbox_world = torch.from_numpy(self.object["actor"][1].get_first_collision_mesh(to_world_frame=True).bounding_box_oriented.extents.copy())
            
            # stats to track
            self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32)
            self.episode_stats = dict(
                # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
                # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
                is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool),
                # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
                consecutive_grasp=torch.zeros((b,), dtype=torch.bool),
                pos_src=torch.zeros((b, 3), dtype=torch.float32),
                pos_tgt=torch.zeros((b, 3), dtype=torch.float32),
            )

    def _settle(self, t: int = 0.5):
        """run the simulation for some steps to help settle the objects"""
        sim_steps = int(self.sim_freq * t / self.control_freq)
        for _ in range(sim_steps):
            self.scene.step()

    def set_initial_qpos(self, env_idx):
        if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam" or self.robot_uids == "panda_bridgedataset_flat_table":
            gripper_pos = 0.04
            robot_init_qpos_noise = 0
            b = len(env_idx)
            # fmt: off
            qpos = np.array(
                [
                    -0.1788885,
                    -0.5299233,
                    0.21601543,
                    -2.9509537,
                    0.16559684,
                    2.4244094,
                    0.6683393,
                    gripper_pos,
                    gripper_pos,
                ]
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
        qpos[:, -2:] = gripper_pos
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    def _get_obs_extra(self, info: Dict):
        """Get task-relevant extra observations. Usually defined on a task by task basis"""
        return dict(
            is_src_obj_grasped=info["is_src_obj_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
    
    def evaluate(self):
        """
        Evaluate whether the environment is currently in a success state by returning a dictionary with a "success" key or
        a failure state via a "fail" key
        """
        info = self._evaluate(
            success_require_src_completely_on_target=True,
        )
        return info
    
    def _evaluate(
        self,
        success_require_src_completely_on_target=True,
        z_flag_required_offset=0.02,
        **kwargs,
    ):
        source_object: Actor = self.object["actor"][0]
        target_object: Actor = self.object["actor"][1]
        source_obj_pose = source_object.pose
        target_obj_pose = target_object.pose

        # whether the source object is grasped
        is_src_obj_grasped = self.agent.is_grasping(source_object)
        # if is_src_obj_grasped:
        self.consecutive_grasp += is_src_obj_grasped
        self.consecutive_grasp[is_src_obj_grasped == 0] = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = (
            self.episode_target_obj_bbox_world / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.episode_source_obj_bbox_world / 2

        pos_src = source_obj_pose.p
        pos_tgt = target_obj_pose.p
        offset = pos_src - pos_tgt
        xy_flag = (
            torch.linalg.norm(offset[:, :2], dim=1)
            <= torch.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.005
        )
        z_flag = (offset[:, 2] > 0) & (
            offset[:, 2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2]
            <= z_flag_required_offset
        )
        src_on_target = xy_flag & z_flag

        if success_require_src_completely_on_target:
            # whether the source object is on the target object based on contact information
            contact_forces = self.scene.get_pairwise_contact_forces(
                source_object, target_object
            )
            net_forces = torch.linalg.norm(contact_forces, dim=1)
            src_on_target = src_on_target & (net_forces > 0.05)

        success = src_on_target

        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] | is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] | consecutive_grasp
        )
        self.episode_stats["pos_src"] = pos_src
        self.episode_stats["pos_tgt"] = pos_tgt
        self.episode_stats["tcp_pose"] = self.agent.tcp.pose.p

        return dict(**self.episode_stats, success=success)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.object["actor"][0].pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_src_obj_grasped = info["is_src_obj_grasped"]
        reward += is_src_obj_grasped

        return reward

