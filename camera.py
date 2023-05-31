"""Simulates the camera module."""

# Not using odometry for this sim, very simple physics environment.

class Camera:

    def __init__(self) -> None:
        # direction angle the robot is facing, 0 is forward, +90 degrees is right,
        # -90 degrees is left, and 180 degrees or -180 degrees is back.
        self.dir_angle = 0
        # reward
        self.reward = 0
        # no. of time steps with no forward translation
        self.no_forward = 0
        # good forward translation?
        self.good_forward_move = True

    """Robot takes a step, returns if in 50 time steps only bad translation
    has occurred, as well as forward translation for that step"""
    def step(self, legs_to_move : list) -> tuple:

        # reset reward
        self.reward = 0

        init_dir = self.dir_angle

        # no movement takes place case
        if len(legs_to_move) == 0:
            self.reward = -2
            self.good_forward_move = False
            return (False, self.good_forward_move)

        # Lets say that moving one right leg is positive(+) angle,
        # and moving one left leg is negative(-) angle.
        # Legs 1,2,3,4 are in this order: FR(0), FL(1), BR(2), BL(3)
        for leg in legs_to_move:
            # right legs
            if leg == 0 or leg == 2:
                self.dir_angle += 5

            # left legs
            else:
                self.dir_angle -= 5

        # case of horrible foward translation
        if abs(self.dir_angle) > 30 and abs(init_dir) > 30 and abs(self.dir_angle - init_dir) == 10:
            self.no_forward += 1
            self.reward = -3
            self.good_forward_move = False

        # case of bad foward translation
        elif abs(self.dir_angle) > 30 and abs(self.dir_angle - init_dir) == 10:
            self.no_forward += 1
            self.reward = -2
            self.good_forward_move = False

        # two legs on the same side moving together
        elif abs(self.dir_angle - init_dir) == 10:
            self.no_forward += 1
            self.reward = -1
            self.good_forward_move = False
        
        # case of ideal translation
        elif abs(self.dir_angle - init_dir) < 5:
            self.no_forward = 0
            self.reward = 5
            self.good_forward_move = True

        # good forward translation
        elif abs(self.dir_angle - init_dir) <= 5:
            self.no_forward = 0
            self.reward = 2
            self.good_forward_move = True

        # case of course correction
        elif abs(init_dir) < 30:
            self.no_forward = 0
            self.reward = 1
            self.good_forward_move = True
        
        # all other cases (between bad and good forward translation)
        else:
            self.no_forward = 0
            self.reward = 0
            self.good_forward_move = True

        # terminate episode if no forward transation occuring
        if self.no_forward == 50:
            print("Terminate Episode- No forward translation in 50 time steps")
            return (True, self.good_forward_move)

        return (False, self.good_forward_move)
