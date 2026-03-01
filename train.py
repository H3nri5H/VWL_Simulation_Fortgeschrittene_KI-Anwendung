import os
import sys
import json
import yaml
import warnings
import argparse
import shutil
import time
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from env.economy_env import SimpleEconomyEnv

warnings.filterwarnings('ignore')
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('ray').setLevel(logging.ERROR)
logging.getLogger('ray.tune').setLevel(logging.ERROR)
logging.getLogger('ray.rllib').setLevel(logging.ERROR)


def load_config():
    """Load training configuration from config.yaml"""
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found!")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_latest_checkpoint():
    """Find most recent checkpoint for resume"""
    checkpoint_dir = Path("./checkpoints").absolute()
    if not checkpoint_dir.exists():
        return None, 0
    
    checkpoints = []
    for cp_dir in checkpoint_dir.iterdir():
        if cp_dir.is_dir() and cp_dir.name.startswith('checkpoint_'):
            metadata_file = cp_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    checkpoints.append((os.path.abspath(str(cp_dir)), metadata.get('iteration', 0)))
    
    if not checkpoints:
        return None, 0
    
    checkpoints.sort(key=lambda x: x[1])
    latest_path, latest_iter = checkpoints[-1]
    return latest_path, latest_iter


def safe_rmtree(path, max_retries=3):
    """Remove directory with Windows permission error handling"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                return False
    return False


def suppress_output(func):
    """Decorator to suppress stdout/stderr"""
    def wrapper(*args, **kwargs):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = open(os.devnull, 'w')
        try:
            result = func(*args, **kwargs)
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout, sys.stderr = old_stdout, old_stderr
        return result
    return wrapper


def train(resume=False):
    """Train multi-agent PPO on economy simulation"""
    config = load_config()
    
    env_cfg = config.get('environment', {})
    env_config = {
        'n_firms': env_cfg.get('n_firms', 2),
        'n_households': env_cfg.get('n_households', 10),
        'max_steps': env_cfg.get('max_steps', 100),
    }
    
    n_firms = env_config['n_firms']
    train_cfg = config.get('training', {})
    config_iterations = train_cfg.get('iterations', 50)
    checkpoint_freq = train_cfg.get('checkpoint_frequency', 10)
    
    ppo_cfg = train_cfg.get('ppo', {})
    res_cfg = train_cfg.get('resources', {})
    
    checkpoint_dir = os.path.abspath("./checkpoints")
    metrics_dir = os.path.abspath("./metrics")
    
    print("\n" + "="*60)
    print("  VWL SIMULATION - MULTI-AGENT RL TRAINING")
    print("="*60)
    
    start_iteration = 0
    resume_checkpoint = None
    total_iterations = config_iterations
    
    if resume:
        resume_checkpoint, start_iteration = find_latest_checkpoint()
        if resume_checkpoint:
            total_iterations = start_iteration + config_iterations
            print(f"\nResuming: Iteration {start_iteration} → {total_iterations}")
        else:
            print(f"\nNo checkpoint found. Starting fresh.")
            resume = False
    else:
        print(f"\nFresh training: {config_iterations} iterations")
    
    if not resume:
        safe_rmtree(checkpoint_dir)
        safe_rmtree(metrics_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
    
    print(f"Environment: {n_firms} firms, {env_config['n_households']} households")
    print(f"Policy: {n_firms}")
    print(f"Workers: {res_cfg.get('num_env_runners', 2)}")
    
    if resume and resume_checkpoint:
        @suppress_output
        def load_algo():
            return PPO.from_checkpoint(os.path.abspath(resume_checkpoint))
        algo = load_algo()
        print("Loaded from checkpoint.")
    else:
        env_temp = SimpleEconomyEnv(env_config)
        policies = {f"firm_{i}": (None, env_temp.observation_space, env_temp.action_space, {}) 
                   for i in range(n_firms)}
        
        rllib_config = (
            PPOConfig()
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .environment(env=SimpleEconomyEnv, env_config=env_config)
            .framework("torch")
            .env_runners(num_env_runners=res_cfg.get('num_env_runners', 2), rollout_fragment_length=200)
            .training(
                train_batch_size=ppo_cfg.get('train_batch_size', 4000),
                minibatch_size=ppo_cfg.get('minibatch_size', 256),
                num_epochs=ppo_cfg.get('num_epochs', 10),
                lr=ppo_cfg.get('learning_rate', 3e-4),
                gamma=ppo_cfg.get('gamma', 0.99),
                lambda_=ppo_cfg.get('lambda', 0.95),
                clip_param=ppo_cfg.get('clip_param', 0.2),
            )
            .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id)
            .resources(num_gpus=res_cfg.get('num_gpus', 0))
        )
        
        @suppress_output
        def build_algo():
            return rllib_config.build()
        algo = build_algo()
        print("Algorithm built.")
    
    print("\n" + "-"*60)
    print(f"{'Iter':<6} {'Reward':<12} {'Min':<10} {'Max':<10} {'EpLen':<8}")
    print("-"*60)
    
    for i in range(start_iteration, total_iterations):
        @suppress_output
        def train_step():
            return algo.train()
        result = train_step()
        
        env_runners = result.get('env_runners', {})
        reward_mean = env_runners.get('episode_reward_mean', 0.0)
        reward_min = env_runners.get('episode_reward_min', 0.0)
        reward_max = env_runners.get('episode_reward_max', 0.0)
        episode_len = env_runners.get('episode_len_mean', 0.0)
        
        print(f"{i+1:<6} {reward_mean:<12.2f} {reward_min:<10.2f} {reward_max:<10.2f} {episode_len:<8.0f}")
        
        if (i + 1) % checkpoint_freq == 0 or (i + 1) == total_iterations:
            iteration_dir = os.path.join(metrics_dir, f"iteration_{i+1}")
            os.makedirs(iteration_dir, exist_ok=True)
            
            with open(os.path.join(iteration_dir, "result.json"), 'w') as f:
                json.dump({
                    'training_iteration': i + 1,
                    'env_runners': {
                        'episode_reward_mean': reward_mean,
                        'episode_reward_min': reward_min,
                        'episode_reward_max': reward_max,
                        'episode_len_mean': episode_len,
                    }
                }, f)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i+1:06d}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            @suppress_output
            def save_checkpoint():
                checkpoint_result = algo.save()
                checkpoint_result.checkpoint.to_directory(checkpoint_path)
            save_checkpoint()
            
            metadata = {
                'iteration': i + 1,
                'reward_mean': reward_mean,
                'episode_len_mean': episode_len,
                'timestamp': result.get('timestamp', 0),
                'is_favorite': (i + 1) == total_iterations,
                'checkpoint_path': checkpoint_path,
                'env_config': env_config
            }
            
            with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            marker = "[*]" if (i + 1) == total_iterations else "[+]"
            print(f"  {marker} Checkpoint saved: iteration {i+1}")
    
    print("-"*60)
    print("\nTraining complete.\n")
    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VWL simulation")
    parser.add_argument("--resume", action='store_true', help="Resume from latest checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
