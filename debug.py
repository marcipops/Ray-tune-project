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
from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
# from FinRL.finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from env_stocktrading_np import StockTradingEnv as StockTradingEnv_numpy 
from finrl.meta.env_stock_trading.env_stocktrading_np_test import StockTradingEnv as StockTradingEnvTest
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
# from env import StockTradingEnv as StockTradingEnv_numpy
import ray
from pprint import pprint
from ray.air.integrations.wandb import WandbLoggerCallback
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
import config_params

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
# print(type(train_env_config))
train_env_config = train_env_config.item()
test_env_config = test_env_config.item()
# from ray.tune.registry import register_env
from ray.tune import register_env
from gymnasium.wrappers import EnvCompatibility
env_name = 'StockTrading_train_env'

def reg_env(config):
    return env(config)
register_env(env_name, lambda config: env(train_env_config))

train_env_instance = env(train_env_config)

from ray.rllib.algorithms.ppo import PPOConfig

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment(env_name)
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    # .training(model={"use_lstm":True})
    .training(model={"use_attention":True})
    .evaluation(evaluation_num_workers=1)
)

algo = config.build()  # 2. build the algorithm,

for _ in range(2):
    print(algo.train())  # 3. train it,
print(algo.evaluate())