"""Simulates the gyroscope module."""


# setting up very simple physics environment that does not take into account
# many forces like friction.


class Gyro:

    def __init__(self) -> None:
        # holds the value of the current balance of the robot. Values can range from 
        # -90 degrees to +90 degrees. 0 degrees is optimal value. If current balance 
        # exceed the absolute value of 25 degrees, we consider balance is lost.
        # Robot falls if balance exceeds absolute value of 40 degrees.
        self.current_balance = 0
        # gyro reward
        self.reward = 0
        # on balance?
        self.on_balance = True
        # if greater than 50, we might want to terminate
        self.bad_move = 0

    """Robot takes a step, return if (robot fell, robot balanced)"""
    def step(self, legs_to_move : list) -> tuple:

        # reset reward
        self.reward = 0
        
        # when no movement occurs case is taken care by camera
        if len(legs_to_move) == 0:
            return (False, self.on_balance)

        # Lets say that moving one right leg is positive(+) balance,
        # and moving one left leg is negative(-) balance.
        # Legs 1,2,3,4 are in this order: FR(0), FL(1), BR(2), BL(3)
        for leg in legs_to_move:
            # right legs
            if leg == 0 or leg == 2:
                # gyro values are not linear
                if self.current_balance >= 20:
                    self.current_balance += 15
                else:
                    self.current_balance += 10

            # left legs
            else:
                if self.current_balance >= -20:
                    self.current_balance -= 15
                else:
                    self.current_balance -= 10

        # checking for robot fall, if so terminate this episode
        if abs(self.current_balance) > 60 and self.bad_move > 50:
            self.on_balance = False
            print("Terminate Episode- Robot fell down")
            return (True, self.on_balance)

        # checking for lost balance
        if abs(self.current_balance) > 30:
            self.on_balance = False
            if len(legs_to_move) > 2:
                self.reward = -2

            else:
                self.reward = 2
                self.bad_move = 0

        # balance maintained
        else:
            self.on_balance = True
            self.reward = 5
            self.bad_move = 0

        # robot has not fallen to the ground, balance status
        return (False, self.on_balance)
