import numpy as np # Import numpy, the python math library
import math
from numpy.linalg import norm
from math import exp
from rlgym_sim.utils.math import cosine_similarity
from rlgym_sim.utils import RewardFunction # Import the base RewardFunction class
from rlgym_sim.utils.gamestates import GameState, PlayerData # Import game state stuff
from rlgym_sim.utils.common_values import (BLUE_GOAL_BACK, BLUE_GOAL_CENTER, ORANGE_GOAL_BACK,
                                       ORANGE_GOAL_CENTER, CAR_MAX_SPEED, ORANGE_TEAM, CAR_MAX_SPEED, BALL_MAX_SPEED, ORANGE_TEAM, BLUE_TEAM,)

#  abs(player.car_data.angular_velocity[2])     # This is spinning sideways such as joystick left or right without air rolling
#  abs(player.car_data.angular_velocity[1])     # This is air roll
#  abs(player.car_data.angular_velocity[0])     # This is summersalting, rolling forward

BACK_WALL_Y = 5120
BALL_RADIUS = 92.75
RAMP_HEIGHT = 256

from numba import jit

@jit(nopython=True)
def distance2D(p1, p2):
    return np.linalg.norm(p2 - p1)

@jit(nopython=True)
def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / BALL_MAX_SPEED
            return float(np.dot(norm_pos_diff, norm_vel))

class AlignBallGoal(RewardFunction):
    def __init__(self, defense=0.1, offense=1.0):
        super().__init__()
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc

        # Align player->ball and net->player vectors
        defensive_reward = self.defense * cosine_similarity(ball - pos, pos - protecc)

        # Align player->ball and player->net vectors
        offensive_reward = self.offense * cosine_similarity(ball - pos, attacc - pos)

        return defensive_reward + offensive_reward

class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:

        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array

        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0

class SpeedTowardBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Velocity of our player
        player_vel = player.car_data.linear_velocity

        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)

        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)

        # We will now normalize our pos_diff vector, so that it has a length/magnitude of 1
        # This will give us the direction to the ball, instead of the difference in position
        # Normalizing a vector can be done by dividing the vector by its length
        dir_to_ball = pos_diff / dist_to_ball

        # Use a dot product to determine how much of our velocity is in this direction
        # Note that this will go negative when we are going away from the ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)

        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0

class DistanceToBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)

        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)

        reward = dist_to_ball / 2000

        return reward

class PlayerSpeedReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState) -> float:
        reward = 0
        player_vel = player.car_data.linear_velocity
        player_vel /= 2300 #2300 is the max car speed
        player_vel *= 100
        player_vel = math.sqrt(player_vel)
        reward = player_vel / 10
        return reward

class AirTouchReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        MAX_TIME_IN_AIR = 1.75 # A rough estimate of the maximum reasonable aerial time
        reward = 0
        if player.ball_touched:
            air_time_frac = min(player.air_time, MAX_TIME_IN_AIR) / MAX_TIME_IN_AIR
            height_frac = ball.position[2] / CommonValues.CEILING_Z
            reward = min(air_time_frac, height_frac)
        return reward

class BallTouchScaledByHeight(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        if player.ball_touched:
            reward = 1 - np.sqrt((state.ball.position[2] / 2000) * 100)
        return reward

class BehindBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        if player.car_data.position[1] < state.ball.position[1]:
            reward += 1
        return reward

class BallTouchReward(RewardFunction):
    # Default constructor
    def __init__(self, Beginner: bool):
        super().__init__()
        self.isBeginner = Beginner
        self.ballVelLast = 0  # Initialize the previous ball velocity to 0

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        self.ballVelLast = 0  # Reset ball velocity on game reset

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0

        if self.isBeginner:
            if player.ball_touched:  # Use player data for ball touch check
                reward += 1.0
        else:
            if player.ball_touched:
                # Calculate velocity gain
                current_ball_velocity = np.linalg.norm(state.ball.linear_velocity)  # Magnitude of ball velocity
                velocity_gain = current_ball_velocity - self.ballVelLast
                reward += max(0, velocity_gain)  # Reward only positive velocity gain
                self.ballVelLast = current_ball_velocity  # Update ball velocity

        return reward

class FaceBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(player.car_data.forward(), norm_pos_diff))

class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        car_speed = np.linalg.norm(player.car_data.linear_velocity)
        car_dir = sign(player.car_data.forward().dot(player.car_data.linear_velocity))
        if car_dir < 0:
            car_speed /= -2300

        else:
            car_speed /= 2300
        return min(car_speed, 1)

class BoostReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return math.sqrt(player.boost_amount * 100) / 10 #do this if it returns between 0 - 100

class BoostPickupReward(RewardFunction):
    # Constructor to initialize prevBoost
    def __init__(self):
        super().__init__()
        self.prevBoost = 0  # Store previous boost amount as an instance variable

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        self.prevBoost = 0  # Reset previous boost amount on reset

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0

        # If the player's boost has increased, reward the difference
        if player.boost_amount > self.prevBoost:
            reward = np.sqrt((player.boost_amount - self.prevBoost) * 100) / 10
            if player.car_data.position[1] > -1000:
                reward *= 1.25

        # Update prevBoost for the next step
        self.prevBoost = player.boost_amount
        return reward

class SmallBoostPickupReward(RewardFunction):
    # Constructor to initialize prevBoost
    def __init__(self):
        super().__init__()
        self.prevBoost = 0  # Store previous boost amount as an instance variable

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        self.prevBoost = 0  # Reset previous boost amount on reset

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0

        # If the player's boost has increased, reward the difference
        if player.boost_amount > self.prevBoost:
            if (player.boost_amount - self.prevBoost) < 0.15: #small pad pick up
                reward += 1
                #print("picked up small pad!")
        # Update prevBoost for the next step
        self.prevBoost = player.boost_amount
        return reward

class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        car_speed = np.linalg.norm(player.car_data.linear_velocity)
        car_dir = np.sign(player.car_data.forward().dot(player.car_data.linear_velocity))
        if car_dir < 0:
            car_speed /= -2300

        else:
            car_speed /= 2300
        return min(car_speed, 1)

class AlignmentReward(RewardFunction):
    def __init__(self, align_w=0.75):
        super().__init__()
        self.align_w = align_w

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_pos = state.ball.position
        player_pos = player.car_data.position

        if player.team_num == ORANGE_TEAM:
            goal_vector = np.array(ORANGE_GOAL_BACK) - player_pos
        else:
            goal_vector = np.array(BLUE_GOAL_BACK) - player_pos

        player_to_ball = ball_pos - player_pos
        alignment = cosine_similarity(player_to_ball, goal_vector)

        reward = self.align_w * alignment
        return float(reward)

class BoostLoseReward(RewardFunction):
    def __init__(self, boost_lose_w=1.0):
        super().__init__()
        self.boost_lose_w = boost_lose_w
        self.last_boost_amount = {}

    def reset(self, initial_state: GameState):
        self.last_boost_amount = {
            player.car_id: player.boost_amount
            for player in initial_state.players
        }

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        last_boost = self.last_boost_amount.get(player.car_id, player.boost_amount)
        current_boost = player.boost_amount
        boost_diff = np.sqrt(np.clip(current_boost, 0, 1)) - np.sqrt(np.clip(last_boost, 0, 1))
        self.last_boost_amount[player.car_id] = current_boost

        reward = 0.0
        if boost_diff < 0:
            car_height = player.car_data.position[2]
            penalty = self.boost_lose_w * boost_diff * (1 - car_height / 642.775) #goal height
            reward += penalty
        return float(reward)

class BallHeightReward(RewardFunction):
    def __init__(self, exponent=1):
        # Exponent should be odd so that negative y -> negative reward
        self.exponent = exponent

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            return (
                state.ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)
            ) ** self.exponent
        else:
            return (
                state.inverted_ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)
            ) ** self.exponent

class FlickReward(RewardFunction):
    def __init__(
        self,
        minimum_barrier: float = 400,
        max_vel_diff: float = 400,
        training: bool = True,
    ):
        super().__init__()
        self.min_distance = minimum_barrier
        self.max_vel_diff = max_vel_diff
        self.training = training

    def reset(self, initial_state: GameState):
        pass

    def stable_carry(self, player: PlayerData, state: GameState) -> bool:
        if BALL_RADIUS + 20 < state.ball.position[2] < BALL_RADIUS + 80:
            if (
                abs(
                    np.linalg.norm(
                        player.car_data.linear_velocity - state.ball.linear_velocity
                    )
                )
                <= self.max_vel_diff
            ):
                return True
        return False

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        stable = self.stable_carry(player, state)

        if stable:
            if player.on_ground:
                return reward # no reward for just dribbling
            else:
                if player.has_flip:
                    # small reward for jumping
                    return reward + 2
                elif abs(player.car_data.angular_velocity[0]) > 3: # car is either front flipping, backflipping, or diagonal flipping
                    # print("PLAYER FLICKED!!!")
                    # big reward for flicking
                    return reward + 5

        return reward

class AerialReward(RewardFunction):
    # Default constructor
    def __init__(self):
        self.AirTime = 0
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        self.AirTime = 0

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0

        MAX_TIME_IN_AIR = 1.75 # A rough estimate of the maximum reasonable aerial time
        air_time_frac = min(self.AirTime, MAX_TIME_IN_AIR) / MAX_TIME_IN_AIR
        height_frac = state.ball.position[2] / 2044

        if player.ball_touched:
            reward = min(air_time_frac, height_frac)

        if player.on_ground:
            self.AirTime = 0
        else:
            self.AirTime += 1/15 # 120 fps / tick skip of 8 = 15 updates per second

        return reward

class DribbleReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        BALL_RADIUS = 92.75

        if (state.ball.position[2] > 100
            and BALL_RADIUS + 20 < state.ball.position[2] < BALL_RADIUS + 200
            and abs(abs(player.car_data.position[0]) - abs(state.ball.position[0])) < 150
            and abs(abs(player.car_data.position[1]) - abs(state.ball.position[1])) < 150):
            reward += 0.2

            if(player.ball_touched):
                reward += 0.8

        return reward

class KickoffProximityReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[0] == 0 and state.ball.position[1] == 0:
            player_pos = np.array(player.car_data.position)
            ball_pos = np.array(state.ball.position)
            player_dist_to_ball = np.linalg.norm(player_pos - ball_pos)

            opponent_distances = []
            for p in state.players:
                if p.team_num != player.team_num:
                    opponent_pos = np.array(p.car_data.position)
                    opponent_dist_to_ball = np.linalg.norm(opponent_pos - ball_pos)
                    opponent_distances.append(opponent_dist_to_ball)

            if opponent_distances and player_dist_to_ball < min(opponent_distances):
                return 1
            else:
                return -1
        return 0

class SpeedflipKickoffReward(RewardFunction):
    def __init__(self, goal_speed=0.5):
        super().__init__()
        self.goal_speed = goal_speed

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[0] == 0 and state.ball.position[1] == 0 and player.boost_amount < 2:
                vel = player.car_data.linear_velocity
                pos_diff = state.ball.position - player.car_data.position
                norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
                norm_vel = vel / CAR_MAX_SPEED
                speed_rew = self.goal_speed * max(float(np.dot(norm_pos_diff, norm_vel)), 0.025)
                return speed_rew
        return 0

class ExampleReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        #reward logic here
        return reward
