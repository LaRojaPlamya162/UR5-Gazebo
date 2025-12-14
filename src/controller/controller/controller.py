#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from controller.bc_model import BCPolicy
from controller.reward import RewardFunction
import torch
import numpy as np
import os
import csv
from ament_index_python.packages import get_package_share_directory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import os
class BCController(Node):
    def __init__(self, node_name, model_name):
        super().__init__(node_name)

        package_share_dir = get_package_share_directory('controller')
        model_path = os.path.join(package_share_dir, model_name)
        # ===== Info =====
        self.intital_ur5_pose = None
        self.intital_ball_pose = None
        self.done = []
        self.resetting = False
        # ===== Model =====
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
        # ===== State =====
        self.last_joint_state = None
        self.ball_pos = None   # (x, y, z)
        self.joint_pos = None
        # ===== Transpose ======
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ===== Logging =====
        self.timestep = 0
        self.csv = open("bc_log_v3.csv", "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestep",
            "s1","s2","s3","s4","s5","s6",
            "a1","a2","a3","a4","a5","a6",
            "ball_pos_x","ball_pos_y","ball_pos_z",
            "wrist_3_x", "wrist_3_y", "wrist_3_z",
            "reward",
            "done"
        ])

        # ===== Timer =====
        self.timer = self.create_timer(0.05, self.control_step)  # 20 Hz
        # ===== Joint limit =====
        self.JOINT_LIMITS = [(-6.28, 6.28)] * 6
    def joint_callback(self, msg):
        self.last_joint_state = msg

    def ball_callback(self, msg):
        pos = msg.pose.pose.position
        x, y, z = pos.x, pos.y, pos.z
        self.ball_pos = [
                x,
                y,
                z
            ]
        # Orientation (quaternion)
        ori = msg.pose.pose.orientation
        

        # Linear velocity
        lin = msg.twist.twist.linear
        vx, vy, vz = lin.x, lin.y, lin.z

        # Angular velocity

        self.get_logger().info(
            f"Pos: [{x:.3f}, {y:.3f}, {z:.3f}] | "
        )
    def control_step(self):
        if self.last_joint_state is None or self.ball_pos is None:
            self.get_logger().info(f"Error")
            return
        if not self.tf_buffer.can_transform("base_link", "wrist_3_link", rclpy.time.Time()):
            return
        """if len(self.done) > 0 and self.done[-1] and not self.resetting:
            self.resetting = True
            self.reset_episode()
            return"""
        x,y,z = np.nan, np.nan, np.nan
        
        # ===== Model prediction =====
        name_to_pos = dict(zip(
            self.last_joint_state.name,
            self.last_joint_state.position
        ))
        obs = np.array([
            name_to_pos[j] for j in self.joint_names
        ], dtype=np.float32)
        obs = torch.tensor(obs)
        dt = 0.05
        with torch.no_grad():
            action = self.model(obs).numpy()
            current_pos = obs
            next_pos = current_pos.numpy() + action * dt
        
        for i, (low, high) in enumerate(self.JOINT_LIMITS):
            next_pos[i] = np.clip(next_pos[i], low, high)
        # ===== Publish trajectory =====
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = next_pos.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(dt * 1e9)
        traj.points.append(point)
        self.pub.publish(traj)

        # ===== Get pose =====
        
        try:    
            trans = self.tf_buffer.lookup_transform(
                "base_link",
                "wrist_3_link",
                rclpy.time.Time()
            )

            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
        except Exception as e:
            self.get_logger().warn(str(e))
        
        # ===== Log =====
        self.joint_pos = [x,y,z]
        reward_fn = RewardFunction(self.ball_pos, self.joint_pos, self.timestep)
        reward = reward_fn.reward
        done = reward_fn.done
        self.done.append(done)
        """if self.timestep == 0:
            self.intial_ur5_pose = obs.tolist()
            self.intial_ball_pose = self.ball_pos.copy()"""
        row = (
        [self.timestep] +
        current_pos.tolist()
        #current_pos.numpy().tolist()
        + action.tolist()
        + self.ball_pos
        + self.joint_pos
        + [reward]
        + [done]
        )
        self.writer.writerow(row)
        self.get_logger().info(f"Sent action: {action}")
        self.timestep += 1 

    def reset_episode(self):
        self.reset_ur5_pose()
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /delete_entity...')

        while not self.respawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /spawn_entity...')

        self.reset_ball_pose()
    def reset_ball_pose(self):
        # ===== Delete ball =====
        req = DeleteEntity.Request()
        req.name = 'cricket_ball'
        #self.delete_client.call_async(req)
        future = self.delete_client.call_async(req)
        #rclpy.spin_until_future_complete(self, future)
        future.add_done_callback(self._on_delete_done)

        # ===== Spawn ball ====
        req = SpawnEntity.Request()
        req.name = 'cricket_ball'
        model_path = os.path.expanduser(
        '~/.gazebo/models/cricket_ball/model.sdf'
        )

        with open(model_path, 'r') as f:
            req.xml = f.read()

        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = 0.414
        pose.orientation.w = 1.0

        req.intial_pose = pose
        req.reference_frame = 'world'
        #self.ball_pos = self.intial_ball_pose.copy()
        #self.respawn_client.call_async(req)
        future = self.respawn_client.call_async(req)
        future.add_done_callback(self._on_spawn_done)
        return
    def reset_ur5_pose(self):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = self.intial_ur5_pose
        point.time_from_start.sec = 2  # reset chậm, an toàn

        traj.points.append(point)
        self.pub.publish(traj)

        self.timestep = 0

        self.done = []
    def _on_spawn_done(self, future):
        self.resetting = False
        self.get_logger().info("Ball respawned")
def main(args=None):
    rclpy.init(args=args)
    node = BCController('bc_controller', 'bc_model_v2.pth')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
