#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from my_controller.bc_model import BCPolicy
import torch
import numpy as np
import os
import csv
from ament_index_python.packages import get_package_share_directory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

class BCController(Node):
    def __init__(self, node_name, model_name):
        #print(node_name)
        #print(model_dir)
        package_share_dir = get_package_share_directory('my_controller')
        model_path = os.path.join(package_share_dir, model_name)
        super().__init__(node_name)

        
        self.model = BCPolicy(state_dim = 6, action_dim = 6)
        self.model.load_state_dict(torch.load(model_path, weights_only = True))
        #self.model = torch.jit.load(model_dir)
        self.model.eval()

        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            10
        )

        self.pub = self.create_publisher(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
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

        self.last_joint_state = None
        self.state_log = []
        self.action_log = []
        #self.csv = open("bc_log.csv", "w", newline = "")
        self.csv = open("bc_log_v2.csv", "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow(["s1","s2","s3","s4","s5","s6","a1","a2","a3","a4","a5","a6"])

    def joint_callback(self, msg):
        self.last_joint_state = msg
        self.control_step()

    def control_step(self):
        if self.last_joint_state is None:
            return
        
        obs = np.array(self.last_joint_state.position, dtype = np.float32)
        obs = torch.tensor(obs)
        print(len(obs))
        with torch.no_grad():
            action = self.model(obs).numpy()

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = action.tolist()
        point.time_from_start.sec = 1

        traj.points.append(point)

        self.pub.publish(traj)
        self.state_log.append(self.last_joint_state.position)
        self.action_log.append(action)
        #np.save("states.npy", np.array(self.state_log))
        #np.save("actions.npy", np.array(self.action_log))
        row = list(self.last_joint_state.position) + list(action)
        self.writer.writerow(row)
        self.get_logger().info(f"Sent action: {action}")

def main(args=None):
    rclpy.init(args=args)
    node = BCController('bc_controller','bc_model_v2.pth')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()