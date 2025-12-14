import math
class RewardFunction():
    def __init__(self,ball_pos, joint_pos, timestep):
        super().__init__()
        self.joint_pos = joint_pos
        self.ball_pos = ball_pos
        self.timestep = timestep
        self.reward = reward(self.ball_pos, self.joint_pos, check_if_ball_out_of_table(self.ball_pos), self.timestep)
    
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
def distance_to_nearest_edge(ball_pos):
    return min(abs(ball_pos[0] - 0),math.abs(0.8 - ball_pos[0]), abs(ball_pos[1] - (-0.4)), abs(0.4 - ball_pos[1]))
def check_if_ball_out_of_table(ball_pos):
    if(ball_pos[2] < 0.414 or ball_pos[0] > 0.8 or ball_pos[0] < 0 or ball_pos[1] > 0.4 or ball_pos[1] < -0.4):
        return False
    return True

def reward(ball_pos, wrist_pos, done, timestep):
    # ---- constants ----
    w_hand = 1.0
    w_edge = 5.0
    w_success = 100.0
    w_time = 0.01

    d_hand_ball = euclidean_distance(ball_pos, wrist_pos)
    d_ball_edge = distance_to_nearest_edge(ball_pos)

    r = 0.0

    # Phase 1: approach ball
    r += -w_hand * d_hand_ball

    # Phase 2: push ball toward edge
    r += -w_edge * d_ball_edge

    # Phase 3: success
    if done:
        r += w_success

    # time penalty
    r += -w_time * timestep

    return r