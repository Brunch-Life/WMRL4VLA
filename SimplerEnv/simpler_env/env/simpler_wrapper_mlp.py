import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles, quaternion_to_matrix




class MLPMS3Wrapper:
    def __init__(self, all_args, device_env, extra_seed=0):
        self.args = all_args
        self.gripper_low = -0.01
        self.gripper_high = 0.04

        self.state_gripper_width_old = np.zeros((self.args.num_envs, 1), dtype=np.float32)

        self.num_envs = self.args.num_envs
        self.device = device_env

        env_kwargs = dict(
        num_envs=self.args.num_envs,
        obs_mode="rgb+segmentation",
        control_mode=self.args.control_mode,
        sim_backend="gpu",
        sim_config={
            "sim_freq": 1000,
            "control_freq": 25,
        },
        max_episode_steps=self.args.episode_len,
        sensor_configs={"shader_pack": self.args.shader},
        is_table_green = self.args.is_table_green,
        )
        if self.args.robot_uids is not None:
            env_kwargs["robot_uids"] = tuple(self.args.robot_uids.split(","))

        if self.args.env_id == "TabletopPickPlaceEnv-v1":
            env_kwargs["object_name"] = self.args.object_name
            env_kwargs["container_name"] = self.args.container_name
        elif self.args.env_id == "TabletopPickEnv-v1":
            env_kwargs["object_name"] = self.args.object_name

        self.env: BaseEnv = gym.make(self.args.env_id, **env_kwargs)

        env_reset_options = {
            "reconfigure": True,
        }
        self.env.reset(seed=[self.args.seed * 1000 + i + extra_seed for i in range(self.args.num_envs)], options=env_reset_options)

        # # variables
        self.reward_dense_old = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)  # [B, 1]
        self.tcp_pose_old = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)  # [B, 3]
        self.pos_src_old = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)  # [B, 3]
        self.pos_tgt_old = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)  # [B, 3]
        self.is_src_obj_grasped_old = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # [B, 1]
        self.gripper_width_old = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)  # [B, 1]

        # # constants
        # bins = np.linspace(-1, 1, 256)
        # sim_backend = "gpu" if self.env.unwrapped.device.type == "gpu" else "cpu"
        # device = str( self.env.unwrapped._sim_device)


    def get_state(self) -> np.ndarray:
        tcp_pose = self.env.unwrapped.agent.ee_pose_at_robot_base
        abs_gripper_width = self.state_gripper_width # [B, 1] -0.01 ~ 0.04
        scaled_gripper_width = (abs_gripper_width - self.gripper_low) / (self.gripper_high - self.gripper_low) # [B, 1] -1 ~ 1

        assert scaled_gripper_width.shape == (self.num_envs, 1)
        state_mat = quaternion_to_matrix(tcp_pose.q).cpu().numpy()
        state_mat_6d = state_mat[:, :3, :2].reshape(-1, 6)
        assert state_mat_6d.shape == (self.num_envs, 6)

        state = np.concatenate([tcp_pose.p.cpu().numpy(), 
                            state_mat_6d,
                            scaled_gripper_width], axis=-1)

        assert state.shape == (self.num_envs, 10)

        return state

    def get_reward(self, info) -> torch.Tensor:
        """
        Sparse reward implementation:
        - Grasp success: +10 reward
        - Task completion: +100 reward 
        - Small time penalty: -0.01 per step
        """
        success = info['success']
        is_src_obj_grasped = info['is_src_obj_grasped']

        rewards = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        
        #for first step - initialize old states for tracking state changes
        for env_idx in range(self.num_envs):
            if info['elapsed_steps'][env_idx] == 1:
                self.is_src_obj_grasped_old[env_idx] = is_src_obj_grasped[env_idx]

            reward = torch.zeros(1, dtype=torch.float32, device=self.device)

            # Small time penalty to encourage efficiency
            time_penalty = -0.01
            reward += time_penalty

            # Sparse reward for successful grasp (when object is first grasped)
            if is_src_obj_grasped[env_idx] and not self.is_src_obj_grasped_old[env_idx]:
                grasp_bonus = 10
                reward += grasp_bonus

            # Sparse reward for task completion
            if success[env_idx]:
                success_bonus = 100
                reward += success_bonus

            # Penalty for dropping the object after grasping
            if self.is_src_obj_grasped_old[env_idx] and not is_src_obj_grasped[env_idx] and not success[env_idx]:
                drop_penalty = -10
                reward += drop_penalty

            rewards[env_idx] = reward
            
            # Update old state for next step
            self.is_src_obj_grasped_old[env_idx] = is_src_obj_grasped[env_idx]

        return rewards

    # ORIGINAL DENSE REWARD CODE (COMMENTED OUT FOR REFERENCE):
    # def get_reward(self, info) -> torch.Tensor:
    #     pos_src = info['pos_src']
    #     pos_tgt = info['pos_tgt']
    #     tcp_pose = info['tcp_pose']
    #     gripper_width = torch.from_numpy(self.state_gripper_width).to(self.device)
    #     success = info['success']
    #     is_src_obj_grasped = info['is_src_obj_grasped']

    #     rewards = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        
    #     #for first step
    #     for env_idx in range(self.num_envs):
    #         if info['elapsed_steps'][env_idx] == 1:
    #             self.tcp_pose_old[env_idx] = tcp_pose[env_idx]
    #             self.pos_src_old[env_idx] = pos_src[env_idx]
    #             self.pos_tgt_old[env_idx] = pos_tgt[env_idx]
    #             self.gripper_width_old[env_idx] = gripper_width[env_idx]
    #             self.is_src_obj_grasped_old[env_idx] = is_src_obj_grasped[env_idx]

    #         reward = torch.zeros(1, dtype=torch.float32, device=self.device)

    #         # time penalty
    #         time_penalty = -0.01
    #         reward += time_penalty

    #         # stage reward
    #         if not is_src_obj_grasped[env_idx]:
    #             # stage 1: move to src
    #             dist_to_object = torch.norm(tcp_pose[env_idx] - pos_src[env_idx])
    #             dist_to_object_old = torch.norm(self.tcp_pose_old[env_idx] - self.pos_src_old[env_idx])
    #             reward += 10 *(dist_to_object_old - dist_to_object)

    #             # stage 2: ready to grasp
    #             dist_xy_to_object = torch.norm(tcp_pose[env_idx, :2] - pos_src[env_idx, :2])
    #             dist_xy_to_object_old = torch.norm(self.tcp_pose_old[env_idx, :2] - self.pos_src_old[env_idx, :2])
    #             reward += 5 *(dist_xy_to_object_old - dist_xy_to_object)

    #         #stage 2.5: reward for success grasp time
    #         if is_src_obj_grasped[env_idx] and not self.is_src_obj_grasped_old[env_idx]:
    #             grasp_bonus = 10
    #             reward += grasp_bonus

    #         if is_src_obj_grasped[env_idx]:
    #             #stage 3: move to tgt
    #             dist_to_object = torch.norm(pos_src[env_idx] - pos_tgt[env_idx])
    #             dist_to_object_old = torch.norm(self.pos_src_old[env_idx] - self.pos_tgt_old[env_idx])
    #             reward += 20 * (dist_to_object_old - dist_to_object)

    #         # final target reward and fail penalty
    #         if success[env_idx]:
    #             success_bonus = 100
    #             reward += success_bonus

    #         if self.is_src_obj_grasped_old[env_idx] and not is_src_obj_grasped[env_idx] and not success[env_idx]:
    #             drop_penalty = -10
    #             reward += drop_penalty

    #         rewards[env_idx] = reward

    #     return rewards
            



    # def get_reward(self, info) -> torch.Tensor:
    #     pos_src = info['pos_src']
    #     pos_tgt = info['pos_tgt']
    #     tcp_pose = info['tcp_pose']
    #     success = info['success']
    #     is_src_obj_grasped = info['is_src_obj_grasped']
    #     consecutive_grasp = info['consecutive_grasp']



    #     reward_dense = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
    #     reward_dense_period_1 = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
    #     reward_dense_period_2 = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
    #     reward_sparse = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

    #     for env_idx in range(self.num_envs):

    #         # compute dense reward
    #         offset_tcp_src = pos_src[env_idx] - tcp_pose[env_idx]
    #         offset_src_tgt = pos_tgt[env_idx] - pos_src[env_idx]

    #         offset_tcp_src_norm = torch.norm(offset_tcp_src)
    #         offset_src_tgt_norm = torch.norm(offset_src_tgt)

    #         reward_dense_period_1[env_idx] = 0.5 - offset_tcp_src_norm
    #         reward_dense_period_2[env_idx] = 0.5 - offset_src_tgt_norm
    #         if not is_src_obj_grasped[env_idx]:
    #             reward_dense[env_idx] += reward_dense_period_1[env_idx]
    #         else:
    #             reward_dense[env_idx] += 1 + reward_dense_period_2[env_idx]

    #         #compute sparse reward
    #         if consecutive_grasp[env_idx]:
    #             reward_sparse[env_idx] += 1

    #         if success[env_idx]:
    #             reward_sparse[env_idx] += 10


    #     reward_dense_diff = reward_dense - self.reward_dense_old
    #     self.reward_dense_old = reward_dense

    #     reward = reward_dense_diff + reward_sparse



    #     return reward

# Removed 6D rotation and transformation matrix methods - no longer needed for 7D direct actions


    def _process_action(self, raw_actions: torch.Tensor) -> np.ndarray:
        # Process 7D absolute action: [pos(3), euler(3), gripper(1)] → return 7D directly
        abs_action = raw_actions.cpu().numpy()  # [B, 7]

        abs_pos = abs_action[:, :3]  # [B, 3] - absolute position
        abs_euler = abs_action[:, 3:6]  # [B, 3] - absolute euler angles
        scaled_gripper_width = abs_action[:, 6:7]  # [B, 1] - gripper (-1 to 1)
        
        # Convert gripper from [-1,1] to actual width
        abs_gripper_width = scaled_gripper_width * (self.gripper_high - self.gripper_low) + self.gripper_low # [B, 1]

        # Return 7D action directly for environment
        action = np.concatenate([abs_pos, abs_euler, abs_gripper_width], axis=-1)  # [B, 7]

        return action

    def reset(self, eps_count: int, reconfigure: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
        # options = {}
        # options["obj_set"] = obj_set
        # if same_init:
        #     options["episode_id"] = torch.randint(1000000000, (1,)).expand(self.num_envs).to(self.env.device)  # [B]


        env_reset_options = {
            "reconfigure": reconfigure,
            "episode_id": torch.arange(self.args.num_envs) + eps_count,
        }

        obs, info = self.env.reset(options=env_reset_options)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8).cpu().numpy()


        self.state_gripper_width = obs['agent']['qpos'][:, -1:].cpu().numpy()

        # get state
        state = self.get_state()

        return obs_image, state, info

    def step(self, raw_action) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        action_abs = self._process_action(raw_action)  # [B, 7] np.ndarray - already in final format

        obs, reward, terminated, truncated, info = self.env.step(action_abs)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8).cpu().numpy()
        truncated = truncated.reshape(-1, 1)  # [B, 1]

        self.state_gripper_width = obs['agent']['qpos'][:, -1:].cpu().numpy()

        state = self.get_state()

        # calculate reward
        reward = self.get_reward(info)

        # process episode info
        if truncated.any():
            info["episode"] = {}
            for k in ["is_src_obj_grasped", "consecutive_grasp", "success"]:
                v = [info[k][idx].item() for idx in range(self.num_envs)]
                info["episode"][k] = v

        return obs_image, state, reward, terminated, truncated, info
