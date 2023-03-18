import ray
assert ray.__version__ >='2.2.0', "Please install ray 2.2.0 by doing 'pip install ray[rllib] ray[tune] lz4' , lz4 is for population based tuning"

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Agg')
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.a2c as a2c
import ray.rllib.algorithms.a3c as a3c
import ray.rllib.algorithms.td3 as td3
import ray.rllib.algorithms.ddpg as ddpg
import ray.rllib.algorithms.appo as appo
import datetime
# %matplotlib inline
from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
# from FinRL.finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv as StockTradingEnv_numpy 
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
# from env import StockTradingEnv as StockTradingEnv_numpy
import ray
from pprint import pprint
# from ray.rllib.algorithms.ppo import ppo
# from ray.rllib.algorithms.ddpg import ddpg
# from ray.rllib.algorithms.a2c import a2c
# from ray.rllib.algorithms.ddpg import ddpg,td3
# from ray.rllib.algorithms import ddpg
import ray.rllib.algorithms.ppo as ppo
# from ray.rllib.algorithms.sac import sac
import sys
# sys.path.append("../FinRL-Library")
import os
import itertools
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.registry import register_env
from ray import air
from ray.air import session
import time
import psutil
psutil_memory_in_bytes = psutil.virtual_memory().total
ray._private.utils.get_system_memory = lambda: psutil_memory_in_bytes
from typing import Dict, Optional, Any

from finrl.config_tickers import DOW_30_TICKER
technical_indicator_list = config.INDICATORS

model_name = 'PPO'
env = StockTradingEnv_numpy
ticker_list = DOW_30_TICKER
data_source = 'yahoofinance'
time_interval = '1D'

import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
with open('train.npy', 'rb') as f:
    train_env_config = np.load(f,allow_pickle=True)

with open('test.npy', 'rb') as f:
    test_env_config = np.load(f,allow_pickle=True)
train_env_config = train_env_config.item()
test_env_config = test_env_config.item()
# from ray.tune.registry import register_env
from ray.tune import register_env
from gymnasium.wrappers import EnvCompatibility
env_name = 'StockTrading_train_env'

def reg_env(config):
    return env(config)
register_env(env_name, lambda : env(train_env_config))

train_env_instance = EnvCompatibility(env(train_env_config))

# %load_ext autoreload
# %autoreload 2

from drllibv2 import DRLlibv2


def sample_ppo_params():
  return {
    #   "entropy_coeff": tune.loguniform(0.00000001, 0.1),
    #   "lr": tune.loguniform(5e-5, 0.001),
    #   "sgd_minibatch_size": tune.choice([ 32, 64, 128, 256, 512]),
    #   "lambda": tune.choice([0.1,0.3,0.5,0.7,0.9,1.0]),
     "entropy_coeff": 0.0000001,
      "lr": 5e-5,
      "sgd_minibatch_size": 64,
      "lambda":0.9,
      "framework":'torch',
      'model':{
        'use_lstm':True,
        # 'lstm_cell_size':tune.choice([128,256,512])
        'lstm_cell_size':256
      },
  }

search_alg = OptunaSearch(
        metric="episode_reward_mean",
    mode="max")
scheduler_ = ASHAScheduler(
        metric="episode_reward_mean",
        mode="max",
        max_t=5,
        grace_period=1,
        reduction_factor=2,
    )

import config_params


drl_agent = DRLlibv2(
    trainable=model_name,
    train_env=env(train_env_config),
    train_env_name="StockTrading_train",
    framework="torch",
    num_workers=config.num_workers,
    log_level="WARN",
    run_name = 'FINRL_TEST_LSTM',
    local_dir = "FINRL_TEST_LSTM",
    params = sample_ppo_params(),
    num_samples = config_params.num_samples,
    num_gpus=config_params.num_gpus,
    training_iterations=config_params.training_iterations,
    checkpoint_freq=config_params.checkpoint_freq,
    # scheduler=scheduler_,
    # search_alg=search_alg
)

mlp_res = drl_agent.train_tune_model()

results_df, best_result = drl_agent.infer_results()

results_df.to_csv("LSTM.csv")

test_env_instance = env(test_env_config)

test_agent = drl_agent.get_test_agent(test_env_instance,'StockTrading_testenv')

obs = test_env_instance.reset()
lstm_cell_size = res.get_best_result().config['model']['lstm_cell_size']
init_state = state = [
    np.zeros([lstm_cell_size], np.float32) for _ in range(2)]
episode_returns = list()  # the cumulative_return / initial_account
episode_total_assets = list()
episode_total_assets.append(test_env_instance.initial_total_asset)
done = False
while not done:
    action, state, _  = test_agent.compute_single_action(observation=obs,state=state)
    obs, reward, done, _ = test_env_instance.step(action)
    # print(action)
    total_asset = (
        test_env_instance.amount
        + (test_env_instance.price_ary[test_env_instance.day] * test_env_instance.stocks).sum()
    )
    episode_total_assets.append(total_asset)
    episode_return = total_asset / test_env_instance.initial_total_asset
    episode_returns.append(episode_return)

import pickle

with open('LSTM', 'wb') as fp:
    pickle.dump(episode_returns, fp)