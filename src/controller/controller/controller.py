#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
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

class BCController(Node):
    def __init__(self, node_name, model_name):
        super().__init__(node_name)

        package_share_dir = get_package_share_directory('controller')
        model_path = os.path.join(package_share_dir, model_name)

        # ===== Model =====
        self.model = BCPolicy(state_dim=6, action_dim=6)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        # ===== Subscriptions =====
        self.ball_name = "my_ball"
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
            "s1","s2","s3","s4","s5","s6",
            "a1","a2","a3","a4","a5","a6",
            "ball_pos_x","ball_pos_y","ball_pos_z",
            "wrist_3_x", "wrist_3_y", "wrist_3_z",
            #"timestep",
            #"reward"
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
        x,y,z = np.nan, np.nan, np.nan
        
        # ===== Model prediction =====
        name_to_pos = dict(zip(
            self.last_joint_state.name,
            self.last_joint_state.position
        ))
        obs = np.array([
            name_to_pos[j] for j in self.joint_names
        ], dtype=np.float32)
        #obs = np.array(self.last_joint_state.position, dtype = np.float32)
        obs = torch.tensor(obs)
        dt = 0.05
        #print(len(obs))
        with torch.no_grad():
            action = self.model(obs).numpy()
            #dt = 0.05
            current_pos = obs
            next_pos = current_pos.numpy() + action * dt
        

        for i, (low, high) in enumerate(self.JOINT_LIMITS):
            next_pos[i] = np.clip(next_pos[i], low, high)
        # ===== Publish trajectory =====
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = next_pos.tolist()
        #point.positions = action.tolist()
        #point.time_from_start.sec = 1
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
        reward = RewardFunction(self.ball_pos, self.joint_pos.numpy().tolist(), self.timestep)
        row = (
        current_pos.numpy().tolist()
        + action.tolist()
        + self.ball_pos
        + self.joint_pos.numpy().tolist()
        #+ self.timestep.numpy().tolist() 
        #+ reward
        )
        self.writer.writerow(row)
        self.get_logger().info(f"Sent action: {action}")
        self.timestep += 1 
        """if (x,y,z != None,None,None):
            self.get_logger().info(
                f"wrist_3_link pose: x={x:.3f}, y={y:.3f}, z={z:.3f}"
            )"""

def main(args=None):
    rclpy.init(args=args)
    node = BCController('bc_controller', 'bc_model_v2.pth')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
