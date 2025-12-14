import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates

class BallPoseListener(Node):
    def __init__(self):
        super().__init__('ball_pose_listener')

        self.subscription = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.callback,
            10
        )

        self.ball_name = "my_ball"

    def callback(self, msg: ModelStates):
        if self.ball_name in msg.name:
            index = msg.name.index(self.ball_name)
            pose = msg.pose[index]

            x = pose.position.x
            y = pose.position.y
            z = pose.position.z

            self.get_logger().info(f"Ball position: x={x}, y={y}, z={z}")

def main(args=None):
    rclpy.init(args=args)
    node = BallPoseListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
