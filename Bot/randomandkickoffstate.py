from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.math import rand_vec3
import numpy as np
from numpy import random as rand

X_MAX = 7000
Y_MAX = 9000
Z_MAX_BALL = 1850
Z_MAX_CAR = 1900
PITCH_MAX = np.pi/2
YAW_MAX = np.pi
ROLL_MAX = np.pi


class RandomState(StateSetter):

    def __init__(self, ball_rand_speed: bool = True, cars_rand_speed: bool = True, cars_on_ground: bool = False):
        """
        RandomState constructor.

        :param ball_rand_speed: Boolean indicating whether the ball will have a randomly set velocity.
        :param cars_rand_speed: Boolean indicating whether cars will have a randomly set velocity.
        :param cars_on_ground: Boolean indicating whether cars should only be placed on the ground.
        """
        super().__init__()
        self.ball_rand_speed = ball_rand_speed
        self.cars_rand_speed = cars_rand_speed
        self.cars_on_ground = cars_on_ground

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to contain random values the ball and each car.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self._reset_ball_random(state_wrapper, self.ball_rand_speed)
        self._reset_cars_random(state_wrapper, self.cars_on_ground, self.cars_rand_speed)

    def _reset_ball_random(self, state_wrapper: StateWrapper, random_speed: bool):
        """
        Function to set the ball to a random position.

        :param state_wrapper: StateWrapper object to be modified.
        :param random_speed: Boolean indicating whether to randomize velocity values.
        """
        state_wrapper.ball.set_pos(rand.random(
        ) * X_MAX - X_MAX/2, rand.random() * Y_MAX - Y_MAX/2, rand.random() * Z_MAX_BALL + 100)
        if random_speed:
            state_wrapper.ball.set_lin_vel(*rand_vec3(3000))
            state_wrapper.ball.set_ang_vel(*rand_vec3(6))

    def _reset_cars_random(self, state_wrapper: StateWrapper, on_ground: bool, random_speed: bool):
        """
        Function to set all cars to a random position.

        :param state_wrapper: StateWrapper object to be modified.
        :param on_ground: Boolean indicating whether to place cars only on the ground.
        :param random_speed: Boolean indicating whether to randomize velocity values.
        """
        for car in state_wrapper.cars:
            # set random position and rotation for all cars based on pre-determined ranges
            car.set_pos(rand.random() * X_MAX - X_MAX/2, rand.random()
                        * Y_MAX - Y_MAX/2, rand.random() * Z_MAX_CAR + 150)
            car.set_rot(rand.random() * PITCH_MAX - PITCH_MAX/2, rand.random()
                        * YAW_MAX - YAW_MAX/2, rand.random() * ROLL_MAX - ROLL_MAX/2)

            car.boost = rand.random()

            if random_speed:
                # set random linear and angular velocity based on pre-determined ranges
                car.set_lin_vel(*rand_vec3(2300))
                car.set_ang_vel(*rand_vec3(5.5))

            # 100% of cars will be set on ground if on_ground == True
            # otherwise, 50% of cars will be set on ground
            if on_ground or rand.random() < 0.5:
                # z position (up/down) is set to ground
                car.set_pos(z=17)
                # z linear velocity (vertical) set to 0
                car.set_lin_vel(z=0)
                # pitch (front of car up/down) set to 0
                # roll (side of car up/down) set to 0
                car.set_rot(pitch=0, roll=0)
                # x angular velocity (affects pitch) set to 0
                # y angular velocity (affects) roll) set to 0
                car.set_ang_vel(x=0, y=0)


class DefaultState(StateSetter):

    SPAWN_BLUE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]

    SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                      0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]

    SPAWN_ORANGE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                        [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]

    SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 *
                        np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected default kickoff.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        # possible kickoff indices are shuffled
        spawn_inds = [0, 1, 2, 3, 4]
        rand.shuffle(spawn_inds)

        blue_count = 0
        orange_count = 0
        for car in state_wrapper.cars:
            pos = [0,0,0]
            yaw = 0
            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = self.SPAWN_BLUE_POS[spawn_inds[blue_count]]
                yaw = self.SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                blue_count += 1
            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = self.SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                yaw = self.SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                orange_count += 1
            # set car state values
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33

class CombinedState(StateSetter):
    def __init__(self, ball_rand_speed: bool = False, cars_rand_speed: bool = False, cars_on_ground: bool = True, weight: float = 0.5):
        """
        CombinedState constructor.

        :param ball_rand_speed: Boolean indicating whether the ball will have a randomly set velocity.
        :param cars_rand_speed: Boolean indicating whether cars will have a randomly set velocity.
        :param cars_on_ground: Boolean indicating whether cars should only be placed on the ground.
        """
        super().__init__()
        self.random_state = RandomState(ball_rand_speed, cars_rand_speed, cars_on_ground)
        self.default_state = DefaultState()
        self.weight = weight

    def reset(self, state_wrapper: StateWrapper):
        """
        Randomly selects between RandomState and DefaultState, ensuring both are used 50% of the time.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        if rand.random() < self.weight: # less than 1 will always happen, always using random state, the lower this number, the more it will use kickoffs
            # Use RandomState
            self.random_state.reset(state_wrapper)
        else:
            # Use DefaultState
            self.default_state.reset(state_wrapper)
