#!/usr/bin/env python3
from controller.models.BC.bc_model import BCPolicy
from controller.components.reward import RewardFunction
import torch
import numpy as np
import os
import csv
import random
from controller.models.SAC.network import Actor,Critic
from controller.models.SAC.sac_model import SACAgent
from controller.models.SAC.replay_buffer import ReplayBuffer
# ===== ROS2 Lib =====
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
class Controller(Node):
    def __init__(self, node_name, model_name):
        super().__init__(node_name)
        package_share_dir = get_package_share_directory('controller')
        model_path = os.path.join(package_share_dir, model_name)
       
        # ===== Info =====
       
        self.done_flag = False
        self.resetting = False
        self.episode_step = 0
        self.episode = 0
        self.timestep = 0
        self.new_episode_ready = True
        self.state_dim = 6
        self.action_dim = 6
        self.batch_size = 256
        self.warmup_steps = 500
        self.total_steps = 0
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_done = None
        self.wait_count = 0
        # ===== Model =====
        self.replay_buffer = ReplayBuffer(capacity = 1000)
        self.agent = SACAgent(state_dim=6, action_dim=6)

        if os.path.exists("SAC.model"):
            self.agent.load_checkpoint("SAC.model")

        self.model = BCPolicy(state_dim=6, action_dim=6)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        # ===== Subscriptions =====
       
        self.ball_name = "cricket_ball"
        self.ball_sub = self.create_subscription(
            Odometry,
            '/cricket_ball/odom',
            self.ball_callback,
            10
        )
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        # ===== Publisher =====
       
        self.pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
       
        # ====== Delete client =====
       
        self.delete_client = self.create_client(
            DeleteEntity,
            '/delete_entity'
        )
        # ===== Respawn client =====
        self.respawn_client = self.create_client(
            SpawnEntity,
            '/spawn_entity'
        )
        self.set_state_client = self.create_client(
            SetEntityState,
            '/gazebo/set_entity_state'
        )
        # ===== State =====
       
        self.last_joint_state = None
        self.ball_pos = None # (x, y, z)
        self.joint_pos = None
        self.intial_ur5_pose = None
        self.intial_ball_pose = None
        # ===== Transpose ======
       
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # ===== Logging =====
       
        self.timestep = 0
        self.episode = 0
        self.csv = open("SAC_log.csv", "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestep",
            "episode",
            "s1","s2","s3","s4","s5","s6",
            "next_s1","next_s2","next_s3","next_s4","next_s5","next_s6",
            "a1","a2","a3","a4","a5","a6",
            "ball_pos_x","ball_pos_y","ball_pos_z",
            "wrist_3_x", "wrist_3_y", "wrist_3_z",
            "reward",
            "done",
        ])
        # ===== Timer =====
        self.timer = self.create_timer(0.05, self.control_step) # 20 Hz
        # ===== Joint limit =====
        self.JOINT_LIMITS = [(-6.28, 6.28)] * 6
    def joint_callback(self, msg):
        self.last_joint_state = msg
    def ball_callback(self, msg):
        pos = msg.pose.pose.position
        x, y, z = pos.x, pos.y, pos.z
        self.ball_pos = [x,y,z]
    def control_step(self):
        if self.resetting or not self.new_episode_ready:
            return
        if self.last_joint_state is None or self.ball_pos is None:
            return
        if not self.tf_buffer.can_transform(
            "base_link", "wrist_3_link", rclpy.time.Time()
        ):
            return
        # ===== Build current state s_t =====
        name_to_pos = dict(zip(
            self.last_joint_state.name,
            self.last_joint_state.position
        ))
        obs = np.array(
            [name_to_pos[j] for j in self.joint_names],
            dtype=np.float32
        )
        # ===== FIRST FRAME =====
        if self.prev_state is None:
            self.prev_state = obs.copy()
            if self.intial_ur5_pose is None:
                self.intial_ur5_pose = obs.copy().tolist()
                self.get_logger().info("Initial UR5 pose captured")
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = self.agent.actor.sample(obs_tensor)
                action = action.numpy()[0]

            '''obs_tensor = torch.from_numpy(obs)
            with torch.no_grad():
                action = self.model(obs_tensor).numpy()'''
            self.prev_action = action.copy()
            self.prev_reward = 0.0
            self.prev_done = False
            self._publish_action(obs, action)
            return
        try:
            trans = self.tf_buffer.lookup_transform(
                "base_link",
                "wrist_3_link",
                rclpy.time.Time()
            )
            ee_pos = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ]
        except Exception:
            ee_pos = [np.nan, np.nan, np.nan]
        self.joint_pos = ee_pos
        # reward for prev action
        reward_fn = RewardFunction(
            self.ball_pos,
            self.joint_pos,
            self.episode_step
        )
        reward = reward_fn.reward
        done = reward_fn.done
        # ===== LOG (s_t, a_t, r_t, s_{t+1}) =====
        row = (
            [self.timestep]
            + [self.episode]
            + self.prev_state.tolist()
            + obs.tolist()
            + self.prev_action.tolist()
            + self.ball_pos
            + self.joint_pos
            + [reward]
            + [done]
        )
        self.writer.writerow(row)
    
        '''obs_tensor = torch.from_numpy(obs)
        with torch.no_grad():
            action = self.model(obs_tensor).numpy()'''

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # (1, state_dim)

        with torch.no_grad():
            action_tensor, _ = self.agent.actor.sample(obs_tensor)
            action = action_tensor.cpu().numpy()[0]

        self._publish_action(obs, action)
        self.prev_state = obs.copy()
        self.prev_action = action.copy()
        self.prev_reward = reward
        self.prev_done = done
        self.timestep += 1
        self.episode_step += 1
        self.replay_buffer.push(self.prev_state,self.prev_action, reward, obs, done)
        if len(self.replay_buffer) >= 1000:
            self.agent.update(self.replay_buffer)
            self.agent.save_checkpoinnnnt("SAC.pth")
            self.replay_buffer = ReplayBuffer(capacity = 1000)
        # ===== Episode end =====
        if done:
            self.get_logger().info(f"Episode {self.episode} done → reset")
            self.resetting = True
            self.new_episode_ready = False
            self.reset_episode()
            return

    def _publish_action(self, obs, action):
        dt = 0.05
        target_pos = obs + action * dt
        for i, (low, high) in enumerate(self.JOINT_LIMITS):
            target_pos[i] = np.clip(target_pos[i], low, high)
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = target_pos.tolist()
        point.time_from_start.nanosec = int(dt * 1e9)
        traj.points.append(point)
        self.pub.publish(traj)
   
    # ===== End of episode reset ====
    def reset_episode(self):
        self.get_logger().info(f"Resetting episode {self.episode + 1}")
        self.done_flag = False
        self.episode += 1
        self.episode_step = 0
        self.new_episode_ready = False
        self.reset_ur5_pose()
        #rclpy.spin_once(self, timeout_sec=2.0)
        self.get_logger().info(f"Start respawning ball pose")
        self.reset_ball_pose()
    def reset_ur5_pose(self):
        if self.intial_ur5_pose is None:
            return
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
       
        point = JointTrajectoryPoint()
        point.positions = self.intial_ur5_pose
        point.time_from_start.sec = 10
        traj.points.append(point)
        self.pub.publish(traj)
        self.new_episode_ready = False
        #self.resetting = True
    def is_ur5_at_initial_pose(self, tol=1e-4):
        if self.last_joint_state is None:
            return False
        if self.intial_ur5_pose is None:
            return False
        name_to_pos = dict(zip(
            self.last_joint_state.name,
            self.last_joint_state.position
        ))
        current = np.array([name_to_pos[j] for j in self.joint_names])
        target = np.array(self.intial_ur5_pose)
        return np.allclose(current, target, atol=tol)
   
    def reset_ball_pose(self):
        # ===== Delete ball =====
        delete_req = DeleteEntity.Request()
        delete_req.name = 'cricket_ball'
        self.get_logger().info(f"Deleting ball...")
        future = self.delete_client.call_async(delete_req)
        future.add_done_callback(self._on_delete_done)
        self.ball_pos = None
      
    def _on_delete_done(self, future):
        self.get_logger().info("Spawning ball...")
        spawn_req = SpawnEntity.Request()
        spawn_req.name = 'cricket_ball'
        model_path = os.path.expanduser(
        '~/.gazebo/models/cricket_ball/model.sdf'
        )
        with open(model_path) as f:
            spawn_req.xml = f.read()
        pose = Pose()
        pose.position.x = random.uniform(0.1, 0.7)
        pose.position.y = random.uniform(-0.4, 0.4)
        pose.position.z = 0.414
        pose.orientation.w = 1.0
        spawn_req.initial_pose = pose
        spawn_req.reference_frame = 'world'
        future = self.respawn_client.call_async(spawn_req)
        future.add_done_callback(self._on_spawn_done)
    def _on_spawn_done(self, future):
        self.get_logger().info("Ball spawned, waiting for UR5...")
        def wait_timer():
            if self.is_ur5_at_initial_pose():
                self.done_flag = False
                self.resetting = False
                self.new_episode_ready = True
                self.prev_state = None
                self.prev_action = None
                self.prev_reward = None
                self.prev_done = None
                self.episode_step = 0
                self.get_logger().info("New episode started")
                self.destroy_timer(timer)
        self.wait_count = 0
        timer = self.create_timer(0.1, wait_timer)
   
def main(args=None):
    rclpy.init(args=args)
    node = Controller('bc_controller', 'bc_model_v2.pth')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()