# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import torch
from isaaclab.utils.math import quat_apply_inverse
##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
# 創建自定義配置
CUSTOM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=300.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.5,
        #     step_height_range=(0.13, 0.2),
        #     step_width=0.25,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.135, 0.135), #0.1 0.13
            step_width=0.46, #棧板深0.46 高 0.13 #樓梯 深0.28
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            vertical_gap=0.1 #0.03 #0.05 #0.07 #0.09
        ),
    },
)


def dual_point_coordinates(
    env: ManagerBasedRLEnvCfg, 
    sensor_cfg: SceneEntityCfg, 
    sensor_cfg_1: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Returns the 3D coordinates of two raycast points in robot body frame."""
    
    # 獲取傳感器
    sensor_0: RayCaster = env.scene.sensors[sensor_cfg.name]
    sensor_1: RayCaster = env.scene.sensors[sensor_cfg_1.name]
    
    # 獲取機器人（使用 SceneEntityCfg 方式）
    robot: RigidObject = env.scene[asset_cfg.name]
    
    # 獲取射線擊中點的世界座標 (取每個傳感器的第一條射線)
    hit_point_0_w = sensor_0.data.ray_hits_w[:, 0, :]  # shape: (num_envs, 3)
    hit_point_1_w = sensor_1.data.ray_hits_w[:, 0, :]  # shape: (num_envs, 3)
    
    # 簡單的數值安全處理
    hit_point_0_w = torch.nan_to_num(hit_point_0_w, nan=0.0, posinf=10.0, neginf=-10.0)
    hit_point_1_w = torch.nan_to_num(hit_point_1_w, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # 獲取機器人的位置和姿態
    robot_pos_w = robot.data.root_pos_w    # shape: (num_envs, 3)
    robot_quat_w = robot.data.root_quat_w  # shape: (num_envs, 4)
    
    # 轉換到本體座標系：P_body = quat_rotate_inverse(quat, P_world - robot_pos)
    hit_point_0_body = quat_apply_inverse(robot_quat_w, hit_point_0_w - robot_pos_w)
    hit_point_1_body = quat_apply_inverse(robot_quat_w, hit_point_1_w - robot_pos_w)
    
    # 限制本體座標系中的範圍 (避免極端值)
    hit_point_0_body = torch.clamp(hit_point_0_body, -5.0, 5.0)
    hit_point_1_body = torch.clamp(hit_point_1_body, -5.0, 5.0)
    
    # 合併兩個點的座標 [x0, y0, z0, x1, y1, z1]
    observation = torch.cat([hit_point_0_body, hit_point_1_body], dim=1)
    
    # 最終安全檢查
    observation = torch.nan_to_num(observation, nan=0.0, posinf=5.0, neginf=-5.0)
    
    return observation

def foot_to_target_reward(
    env: ManagerBasedRLEnvCfg,
    height_scanner_cfg: SceneEntityCfg,        # 第一個height scanner
    height_scanner_1_cfg: SceneEntityCfg,      # 第二個height scanner 
    left_foot_name: str = "FL_foot",
    right_foot_name: str = "FR_foot", 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    xy_scale: float = 1.0,
    z_scale: float = 0.5,
    max_distance: float = 0.5,
    gait_phase_scale: float = 1.0,  # 新增：步態相位獎勵權重
    alternating_reward: bool = True  # 新增：是否啟用交替步態獎勵
) -> torch.Tensor:
    """
    改進的獎勵函數：鼓勵四足動物的自然對稱步態
    
    主要改進：
    1. 基於步態相位的獎勵：鼓勵前腳交替運動
    2. 可選的交替獎勵機制
    3. 動態目標分配：根據當前步態相位決定哪隻腳應該往哪個目標
    """
    
    # 獲取機器人
    robot: RigidObject = env.scene[asset_cfg.name]
    
    # 獲取目標點座標
    target_coords = dual_point_coordinates(
        env=env,
        sensor_cfg=height_scanner_cfg,
        sensor_cfg_1=height_scanner_1_cfg,
        asset_cfg=asset_cfg
    )
    
    # 提取左右目標點
    target_left = target_coords[:, :3]    # [x0, y0, z0]
    target_right = target_coords[:, 3:6]  # [x1, y1, z1]
    
    # 獲取機器人的位置和姿態
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    
    # 獲取左右前腳在世界座標系的位置
    try:
        left_foot_indices = robot.find_bodies(left_foot_name)
        right_foot_indices = robot.find_bodies(right_foot_name)
        
        if len(left_foot_indices) == 0 or len(right_foot_indices) == 0:
            return torch.zeros(target_coords.shape[0], device=target_coords.device)
            
        left_foot_pos_w = robot.data.body_pos_w[:, left_foot_indices[0]].squeeze(1)
        right_foot_pos_w = robot.data.body_pos_w[:, right_foot_indices[0]].squeeze(1)
        
    except Exception as e:
        return torch.zeros(target_coords.shape[0], device=target_coords.device)
    
    # 轉換到機器人本體座標系
    left_foot_body = quat_apply_inverse(robot_quat_w, left_foot_pos_w - robot_pos_w)
    right_foot_body = quat_apply_inverse(robot_quat_w, right_foot_pos_w - robot_pos_w)
    
    # 方案1: 基於時間的步態相位 (推薦)
    if alternating_reward:
        # 使用仿真時間創建步態相位 (可調整頻率)
        time_step = env.step_dt if hasattr(env, 'step_dt') else 0.005
        current_time = getattr(env, 'episode_length_s', torch.zeros_like(target_coords[:, 0])) 
        
        # 創建步態週期 (例如：1秒一個週期)
        gait_frequency = 1.5  # Hz，可根據需要調整
        phase = (current_time * gait_frequency) % 1.0  # [0, 1)
        
        # 動態分配目標：
        # phase < 0.5: 左腳往左目標，右腳往右目標
        # phase >= 0.5: 左腳往右目標，右腳往左目標
        use_original_assignment = (phase < 0.5).float().unsqueeze(1)
        
        # 左腳的目標
        left_target = use_original_assignment * target_left + (1 - use_original_assignment) * target_right
        # 右腳的目標  
        right_target = use_original_assignment * target_right + (1 - use_original_assignment) * target_left
        
    else:
        # 原始固定分配
        left_target = target_left
        right_target = target_right
    
    # 計算距離
    left_distance_vec = left_foot_body - left_target
    right_distance_vec = right_foot_body - right_target
    
    # XY和Z方向距離
    left_xy_dist = torch.norm(left_distance_vec[:, :2], dim=1)
    left_z_dist = torch.abs(left_distance_vec[:, 2])
    
    right_xy_dist = torch.norm(right_distance_vec[:, :2], dim=1)
    right_z_dist = torch.abs(right_distance_vec[:, 2])
    
    # 加權總距離
    left_total_dist = xy_scale * left_xy_dist + z_scale * left_z_dist
    right_total_dist = xy_scale * right_xy_dist + z_scale * right_z_dist
    
    # 基礎獎勵（指數衰減）
    left_reward = torch.exp(-left_total_dist / max_distance)
    right_reward = torch.exp(-right_total_dist / max_distance)
    
    # 距離限制
    left_reward = torch.where(left_total_dist < max_distance, left_reward, torch.zeros_like(left_reward))
    right_reward = torch.where(right_total_dist < max_distance, right_reward, torch.zeros_like(right_reward))
    
    # 額外的步態相位獎勵：鼓勵交替運動
    if alternating_reward:
        # 計算腳部相對位置差異（鼓勵非同步運動)
        foot_diff = torch.norm(left_foot_body - right_foot_body, dim=1)
        
        # 當兩腳距離適中時給予額外獎勵（避免完全同步）
        optimal_foot_separation = 0.3  # 可調整
        separation_reward = torch.exp(-torch.abs(foot_diff - optimal_foot_separation) / 0.1)
        
        # 結合基礎獎勵和步態獎勵
        total_reward = (left_reward + right_reward) / 2.0 + gait_phase_scale * separation_reward
    else:
        total_reward = (left_reward + right_reward) / 2.0
    
    # 速度條件限制
    x_vel_command = env.command_manager.get_command(command_name)[:, 0]
    total_reward *= (x_vel_command > 0.2).float()
    
    return total_reward
        
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=CUSTOM_TERRAINS_CFG, #ROUGH_TERRAINS_CFG CUSTOM_TERRAINS_CFG
        max_init_terrain_level=5, #5 #10
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.14, 0.15)),
        attach_yaw_only=False,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=1.0,
            size=[0.0, 0.0],
        ),
        max_distance=2.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    height_scanner_1 = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", 
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, -0.14, 0.15)),
        attach_yaw_only=False,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=1.0,
            size=[0.0, 0.0],
        ),
        max_distance=2.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

# #Privileged
#     height_scanner_2 = RayCasterCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/base",
#         offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.14, 0.15)),
#         attach_yaw_only=True,
#         pattern_cfg=patterns.GridPatternCfg(
#             resolution=0.1,
#             size=[0.5, 0.5],
#         ),
#         max_distance=2.0,
#         debug_vis=False,
#         mesh_prim_paths=["/World/ground"],
#     )

#     height_scanner_3 = RayCasterCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/base",
#         offset=RayCasterCfg.OffsetCfg(pos=(0.4, -0.14, 0.15)),
#         attach_yaw_only=True,
#         pattern_cfg=patterns.GridPatternCfg(
#             resolution=0.1,
#             size=[0.5, 0.5],
#         ),
#         max_distance=2.0,
#         debug_vis=False,
#         mesh_prim_paths=["/World/ground"],
#     )
    

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        dual_point_coords = ObsTerm(
            func=dual_point_coordinates,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "sensor_cfg_1": SceneEntityCfg("height_scanner_1")
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-2.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # actions = ObsTerm(func=mdp.last_action)
        # height_priv_1 = ObsTerm(
        #     func=mdp.height_scan,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("height_scanner_2")
        #     },
        #     clip=(-2.0, 2.0),
        # )
        # height_priv_2 = ObsTerm(
        #     func=mdp.height_scan,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("height_scanner_3")
        #     },
        #     clip=(-2.0, 2.0),
        # )
        def __post_init__(self):
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*CALF"), "threshold": 1.0},#THIGH
    )
    foot_target_tracking = RewTerm(
        func=foot_to_target_reward,
        weight=1.0, 
        params={
            "height_scanner_cfg": SceneEntityCfg("height_scanner"),
            "height_scanner_1_cfg": SceneEntityCfg("height_scanner_1"),
            "xy_scale": 1.0,
            "z_scale": 0.3,
            "max_distance": 0.1
        }
    )
    
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    
    similar_to_default = RewTerm(func=mdp.similar2default, weight=-0.1) 

    action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class RoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False





#################################################################

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class HeightScanGo2EnvCfg(RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.viewer.eye = (7.0, 7.0, 7.0)

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.4
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 1.0
        
        
        #self.scene.terrain.terrain_generator.sub_terrains["flat"].proportion = 1.0

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0) #(-1.0,3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.8 # resume
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_calf"
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.dof_torques_l2.weight = -0.00002  #-0.00002 
        self.rewards.track_lin_vel_xy_exp.weight = 5.0 #5  #1.5
        self.rewards.track_ang_vel_z_exp.weight = 2.5 #2.5  #0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7 #-7
        
        self.rewards.similar_to_default.weight = -0.35

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class HeightScanGo2EnvCfg_PLAY(HeightScanGo2EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.terminations.time_out=None