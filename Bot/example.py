import numpy as np
from numpy import random as rand
from rlgym_sim.utils.gamestates import GameState
from rlgym_sim.utils.gamestates.player_data import PlayerData
from rlgym_ppo.util import MetricsLogger
import rewards
from rewards import VelocityBallToGoalReward, AlignBallGoal, InAirReward, SpeedTowardBallReward, DistanceToBallReward, PlayerSpeedReward, AirTouchReward, BallTouchScaledByHeight, BehindBallReward, BallTouchReward, FaceBallReward, SpeedReward, BoostReward, BoostPickupReward, SmallBoostPickupReward, SpeedReward, AlignmentReward, BoostLoseReward, BallHeightReward, FlickReward, AerialReward, DribbleReward, KickoffProximityReward, SpeedflipKickoffReward
from zerosumreward import ZeroSumReward

avg_goals_last = 0  # Initialize it outside the class scope (global)

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [
                game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score,  # Orange team's score
                game_state.players[0].boost_amount,  # Player's boost amount (fixed reference)
                game_state.players[0].on_ground,  # Whether the player is on the ground
                game_state.players[0].match_goals,  # Goals scored by the player
                game_state.players[0].match_demolishes,  # Demolitions caused by the player
                game_state.players[0].ball_touched, # Touched the ball
                game_state.ball.linear_velocity,  # Ball's linear velocity
                game_state.players[0].match_saves # Saves
               ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        global avg_goals_last  # Referencing the global variable

        avg_car_vel = 0
        avg_ball_vel = 0
        avg_goals = 0
        avg_goals_increase = 0
        avg_boost = 0
        avg_airtime = 0
        avg_demos = 0
        avg_touches = 0
        avg_saves = 0

        for metric_array in collected_metrics:
            car_velocity = metric_array[0]
            car_vel_magnitude = np.linalg.norm(car_velocity)
            avg_car_vel += car_vel_magnitude

            avg_boost += metric_array[3]
            avg_airtime += 1 - metric_array[4]
            avg_goals += metric_array[5]
            avg_demos += metric_array[6]
            avg_touches += metric_array[7]

            ball_velocity = metric_array[8]
            ball_vel_magnitude = np.linalg.norm(ball_velocity)
            avg_ball_vel += ball_vel_magnitude

            avg_saves += metric_array[9]

        avg_car_vel /= len(collected_metrics)
        avg_ball_vel /= len(collected_metrics)
        avg_boost /= len(collected_metrics)
        avg_airtime /= len(collected_metrics)
        avg_goals /= len(collected_metrics)
        avg_demos /= len(collected_metrics)
        avg_touches /= len(collected_metrics)
        avg_saves /= len(collected_metrics)

        # Ensure avg_goals_last is initialized before using it
        avg_goals_increase = avg_goals - avg_goals_last
        avg_goals_last = avg_goals  # Update avg_goals_last for future use

        avg_boost_big = avg_boost * 1000

        report = {
            "average player speed": avg_car_vel,
            "average ball speed": avg_ball_vel,
            "average boost": avg_boost,
            "average boost 0-1000": avg_boost_big,
            "average airtime": avg_airtime,
            "average goals": avg_goals,
            "average demos": avg_demos,
            "ball touch ratio": avg_touches,
            "Cumulative Timesteps": cumulative_timesteps,
            "average goals increase": avg_goals_increase,
            "average saves": avg_saves
        }

        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
        EventReward
    from lookupact import LookupAction
    from customobs import AdvancedObs
    from rlgym_sim.utils.state_setters.random_state import RandomState
    from rlgym_sim.utils.state_setters.default_state import DefaultState
    from randomandkickoffstate import CombinedState
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values

    spawn_opponents = True
    team_size = 1

    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    obs_builder = AdvancedObs()
                                                         # the below weight of 0.2 means that it will use random state 20% of the time, and kickoffs 80% of the time
    state_setter = CombinedState(True, True, False, 0.2) # F, F, T makes the ball float in the air until someone touches it, T, T, F is normal
                                                         # less than 1 will always happen, always using random state, the lower this number, the more it will use kickoffs
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    # zero sum reward usage  (ZeroSumReward(YourReward(), team_spirit, opp_scale), yourweight)
    # make demo reward = goal reward for it to learn demos
    reward_fn = CombinedReward.from_zipped( # Format is (func, weight)
        # basic rewards
        (InAirReward(), 0.2),
        (FaceBallReward(), 1.0),
        #(KickoffProximityReward(), 15.0), # closer to ball on kickoff than opponent
        (SpeedTowardBallReward(), 5.0),
        #(VelocityBallToGoalReward(), 2.0),
        #(BoostReward(), 0.1),
        #(BoostPickupReward(), 1.33),
        #(SpeedReward(), 0.05),
        (BallTouchReward(True), 50.0),
        #(EventReward(team_goal=1.0, concede=-0.85, demo=0.35), 750.0),
    )


    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)

    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    ts_per_iteration = 50_000
    n_proc = 32
    policy_layer_sizes = (512, 512, 512)
    critic_layer_sizes = (1024, 1024, 512)

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
                      build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      policy_layer_sizes=policy_layer_sizes,
                      critic_layer_sizes=critic_layer_sizes,
                      policy_lr=2e-4,
                      critic_lr=2e-4,
                      ppo_batch_size=ts_per_iteration, # 50k default
                      ts_per_iteration=ts_per_iteration,
                      exp_buffer_size=ts_per_iteration * 3, # exp_buffer_size is how many past steps it trains on, this smooths training
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.01,
                      ppo_epochs=2, # epochs is how many times the bot trains on the same batch of steps
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=50_000_000,
                      timestep_limit=10_000_000_000,
                      log_to_wandb=True,
                      render = False, # if rendering, change n_proc to 1
                      render_delay = 0.025
                      )
    learner.learn()
