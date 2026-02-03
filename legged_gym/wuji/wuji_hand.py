import wujihandpy
import numpy as np
class wuji_node:
    """硬件薄封装：只在子进程里实例化 & 访问"""
    def __init__(self,serial_number, is_use_csp: bool = True):
        self.hand = wujihandpy.Hand(serial_number = serial_number)
        self.is_use_csp = is_use_csp
        if is_use_csp:
            self.hand.write_joint_control_mode(np.uint16(4))  # CSP
        else:
            self.hand.write_joint_control_mode(np.uint16(2))  # PTP
        self.hand.write_global_tpdo_id(np.uint16(1))
        self.hand.write_pdo_interval(np.uint32(1000))
        self.hand.write_pdo_enabled(np.uint8(1))
        self.hand.write_joint_control_word(np.uint16(1))     # enable
        self.hand.write_joint_current_limit(np.uint16(2000))
        



    def get_pose(self) -> np.ndarray:
        raw = self.hand.read_joint_position()  # (5,4) raw ints
        raw = self.hand.read_joint_position()  # (5,4) raw ints
        raw = self.hand.read_joint_position()  # (5,4) raw ints
        raw = self.hand.read_joint_position()  # (5,4) raw ints

        # raw[1,0] =  -raw[1,0]
        # raw[2,0] =  -raw[2,0]
        # raw[3,0] =  -raw[3,0]
        # raw[4,0] =  -raw[4,0]
        
        return raw.reshape(20,)

    def set_all_joints_pos(self, positions_in_radian: np.ndarray):
        
        # print("set_all_joints_pos", positions_in_radian[[4,8,12,16]])
        # print("before hack:", positions_in_radian)
        assert positions_in_radian.shape == (20,)
        
        # positions_in_radian[4] =  -positions_in_radian[4]
        # positions_in_radian[8] =  -positions_in_radian[8]
        # positions_in_radian[12] =  -positions_in_radian[12] 
        # positions_in_radian[16] =  -positions_in_radian[16]
       

        angles = positions_in_radian.reshape(5, 4)

        if self.is_use_csp:
            self.hand.pdo_write_unchecked(angles)
        else:
            self.hand.write_joint_control_position(angles)

    def csp_set_all_joints_pos(self, positions_in_radian: np.ndarray):
        assert self.is_use_csp
        assert positions_in_radian.shape == (5, 4)
        self.hand.pdo_write_unchecked(positions_in_radian)

    def __del__(self):
        try:
            self.hand.write_joint_control_word(np.uint16(5))  # disable
        except Exception:
            pass

def main():
    # left_hand = wuji_node(serial_number="3375387C3233", is_use_csp=False)
    # left_hand.set_all_joints_pos(np.zeros(20,))
    right_hand = wuji_node(serial_number="337238723233", is_use_csp=False)
    pose = np.array([0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0])
    # pose = np.ones(20)*0.2

    right_hand.set_all_joints_pos(pose)

    import time
    time.sleep(0.5)

 

if __name__ == '__main__':
    main()