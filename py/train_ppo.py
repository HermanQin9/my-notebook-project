\
# train_ppo.py — 训练入口脚本
from stable_baselines3 import PPO
from env import PipelineEnv
import pandas as pd

if __name__ == "__main__":
    df_train = pd.read_csv("data/processed/non_fe.csv")  # TODO: 修改为实际路径
    env = PipelineEnv(df_train)
    model = PPO("MlpPolicy", env, verbose=1, clip_range=0.2, target_kl=0.01)
    model.learn(total_timesteps=10000)
    model.save("models/ppo_pipeline.zip")
    print("PPO model trained and saved.")
