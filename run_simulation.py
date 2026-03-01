import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# CRITICAL: Set these BEFORE importing Ray
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['RAY_COLOR_PREFIX'] = '0'
os.environ['RAY_LOG_TO_STDERR'] = '0'

import warnings
import logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)
for logger_name in ['ray', 'ray.tune', 'ray.rllib', 'ray.serve', 'ray.core']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False

from ray.rllib.algorithms.ppo import PPO
from env.economy_env import SimpleEconomyEnv


class SuppressOutput:
    """Context manager to suppress stdout/stderr"""
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


def load_config():
    """Load environment configuration"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def list_checkpoints():
    """List all available checkpoints"""
    checkpoint_dir = Path("./checkpoints")
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for cp_dir in checkpoint_dir.iterdir():
        if cp_dir.is_dir() and cp_dir.name.startswith('checkpoint_'):
            metadata_file = cp_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    is_favorite = metadata.get('is_favorite', False)
                    iteration = metadata.get('iteration', 0)
                    timestamp = metadata.get('timestamp', 'unknown')
                    checkpoints.append({
                        'path': os.path.abspath(str(cp_dir)),
                        'iteration': iteration,
                        'favorite': is_favorite,
                        'timestamp': timestamp
                    })
    
    checkpoints.sort(key=lambda x: x['iteration'])
    return checkpoints


def select_checkpoint_interactive():
    """Interactive checkpoint selection"""
    checkpoints = list_checkpoints()
    
    if not checkpoints:
        print("Error: No checkpoints found.")
        return None
    
    print("\n" + "="*60)
    print("  AVAILABLE CHECKPOINTS")
    print("="*60)
    for idx, cp in enumerate(checkpoints, 1):
        fav_marker = " ★" if cp['favorite'] else ""
        print(f"{idx}. Iteration {cp['iteration']:3d} - {cp['timestamp']}{fav_marker}")
    print("="*60)
    
    while True:
        try:
            choice = input(f"\nSelect checkpoint [1-{len(checkpoints)}]: ").strip()
            if not choice:
                print("Using latest checkpoint...")
                return checkpoints[-1]['path']
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]['path']
            print(f"Invalid choice. Please enter 1-{len(checkpoints)}.")
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            return None


def get_seed_interactive():
    """Interactive seed selection"""
    while True:
        try:
            seed_input = input("\nEnter seed (press Enter for random): ").strip()
            if not seed_input:
                seed = np.random.randint(0, 1000000)
                print(f"Generated random seed: {seed}")
                return seed
            return int(seed_input)
        except ValueError:
            print("Invalid seed. Please enter a number.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def get_steps_interactive(default_steps):
    """Interactive steps selection"""
    while True:
        try:
            steps_input = input(f"\nEnter number of steps (press Enter for default {default_steps}): ").strip()
            if not steps_input:
                print(f"Using default: {default_steps} steps")
                return default_steps
            steps = int(steps_input)
            if steps > 0:
                return steps
            print("Steps must be positive.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def run_simulation(checkpoint_path=None, seed=None, max_steps=None, interactive=True):
    """Run single simulation episode with trained policies"""
    
    # Interactive mode
    if interactive:
        if checkpoint_path is None:
            checkpoint_path = select_checkpoint_interactive()
            if checkpoint_path is None:
                return
        
        if seed is None:
            seed = get_seed_interactive()
            if seed is None:
                return
        
        # Load config to get default steps
        config = load_config()
        default_steps = config.get('environment', {}).get('max_steps', 365)
        
        if max_steps is None:
            max_steps = get_steps_interactive(default_steps)
            if max_steps is None:
                return
    else:
        # Non-interactive fallbacks
        if checkpoint_path is None:
            checkpoints = list_checkpoints()
            if not checkpoints:
                print("Error: No checkpoints found.")
                return
            checkpoint_path = checkpoints[-1]['path']
        
        if seed is None:
            seed = np.random.randint(0, 1000000)
        
        if max_steps is None:
            config = load_config()
            max_steps = config.get('environment', {}).get('max_steps', 365)
    
    # Load algorithm
    with SuppressOutput():
        algo = PPO.from_checkpoint(checkpoint_path)
    
    metadata_file = Path(checkpoint_path) / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            env_config = metadata.get('env_config', {})
            iteration = metadata.get('iteration', 0)
    else:
        config = load_config()
        env_cfg = config.get('environment', {})
        env_config = {
            'n_firms': env_cfg.get('n_firms', 10),
            'n_households': env_cfg.get('n_households', 3000),
            'max_steps': max_steps,
        }
        iteration = 0
    
    # Override max_steps with user choice
    env_config['max_steps'] = max_steps
    env = SimpleEconomyEnv(env_config)
    
    print("\n" + "="*60)
    print("  VWL SIMULATION - RUNNING EPISODE")
    print("="*60)
    print(f"Checkpoint: iteration {iteration}")
    print(f"Environment: {env_config['n_firms']} firms, {env_config['n_households']} households")
    print(f"Max Steps: {max_steps}")
    print(f"Seed: {seed}")
    print("="*60 + "\n")
    
    obs, _ = env.reset(seed=seed)
    done = {'__all__': False}
    step = 0
    
    firm_history = {f"firm_{i}": [] for i in range(env_config['n_firms'])}
    household_history = []
    
    while not done['__all__']:
        actions = {}
        for agent_id in obs.keys():
            with SuppressOutput():
                actions[agent_id] = algo.compute_single_action(obs[agent_id], policy_id=agent_id)
        
        obs, rewards, dones, _, infos = env.step(actions)
        done = dones
        step += 1
        
        # Record firm data
        for firm_id in firm_history.keys():
            firm = env.firms[firm_id]
            firm_history[firm_id].append({
                'step': step,
                'price': firm['price'],
                'wage': firm['wage'],
                'employees': firm['employees'],
                'inventory': firm['inventory'],
                'production': firm['production'],
                'capital': firm['capital'],
                'profit': firm['profit'],
                'revenue': firm['revenue'],
                'costs': firm['costs'],
                'sales': firm['sales'],
                'quality': firm['quality'],
                'marketing': firm['marketing'],
                'bankrupt': firm['bankrupt'],
            })
        
        for idx, household in enumerate(env.households):
            household_history.append({
                'step': step,
                'household_id': idx,
                'money': household['money'],
                'skill_level': household['skill_level'],
                'max_acceptable_price': household['max_acceptable_price'],
                'employer': household['employer'] if household['employer'] else 'unemployed',
                'wage': household['wage'],
                'wealth_type': household['wealth_type'],
            })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("./simulation_results")
    results_dir.mkdir(exist_ok=True)
    
    import csv
    
    # Save firms data
    firms_file = results_dir / f"firms_checkpoint{iteration}_seed{seed}_{timestamp}.csv"
    with open(firms_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'firm_id', 'step', 'price', 'wage', 'employees', 
                        'inventory', 'production', 'capital', 'profit', 'revenue', 'costs', 
                        'sales', 'quality', 'marketing', 'bankrupt'])
        for firm_id, history in firm_history.items():
            for record in history:
                writer.writerow([seed, firm_id, record['step'], record['price'], 
                               record['wage'], record['employees'], record['inventory'], 
                               record['production'], record['capital'], record['profit'], 
                               record['revenue'], record['costs'], record['sales'], 
                               record['quality'], record['marketing'], record['bankrupt']])
    
    # Save households data
    households_file = results_dir / f"households_checkpoint{iteration}_seed{seed}_{timestamp}.csv"
    with open(households_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'step', 'household_id', 'money', 'skill_level', 
                        'max_acceptable_price', 'employer', 'wage', 'wealth_type'])
        for record in household_history:
            writer.writerow([seed, record['step'], record['household_id'], 
                           record['money'], record['skill_level'], record['max_acceptable_price'], 
                           record['employer'], record['wage'], record['wealth_type']])
    
    # Summary
    survivors = sum(1 for f in env.firms.values() if not f['bankrupt'])
    avg_capital = np.mean([f['capital'] for f in env.firms.values() if not f['bankrupt']]) if survivors > 0 else 0
    total_household_money = sum(hh['money'] for hh in env.households)
    avg_household_money = total_household_money / len(env.households)
    
    # Count segment distribution
    segments = {'budget': 0, 'mainstream': 0, 'premium': 0}
    for firm in env.firms.values():
        if not firm['bankrupt']:
            if firm['price'] < 40:
                segments['budget'] += 1
            elif firm['price'] > 70 or firm['quality'] > 1.5:
                segments['premium'] += 1
            else:
                segments['mainstream'] += 1
    
    summary_file = results_dir / f"summary_seed{seed}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Checkpoint: iteration {iteration}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Max Steps: {max_steps}\n")
        f.write(f"\n--- FIRMS ---\n")
        f.write(f"Survivors: {survivors}/{env_config['n_firms']}\n")
        f.write(f"Average Capital: {avg_capital:.2f}\n")
        f.write(f"\nSegment Distribution:\n")
        f.write(f"  Budget: {segments['budget']}\n")
        f.write(f"  Mainstream: {segments['mainstream']}\n")
        f.write(f"  Premium: {segments['premium']}\n")
        f.write(f"\n--- HOUSEHOLDS ---\n")
        f.write(f"Total Households: {env_config['n_households']}\n")
        f.write(f"Average Money: {avg_household_money:.2f}\n")
        f.write(f"Total Money: {total_household_money:.2f}\n")
    
    print(f"\nSimulation complete!")
    print(f"Survivors: {survivors}/{env_config['n_firms']}")
    print(f"Segments: Budget={segments['budget']}, Mainstream={segments['mainstream']}, Premium={segments['premium']}")
    print(f"Avg Capital: {avg_capital:.2f}\n")
    print(f"Results saved:")
    print(f"  - {firms_file.name}")
    print(f"  - {households_file.name}")
    print(f"  - {summary_file.name}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run VWL simulation with trained RL policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python run_simulation.py

  # Non-interactive with specific parameters
  python run_simulation.py --checkpoint ./checkpoints/checkpoint_000020 --seed 42 --steps 500

  # Non-interactive with defaults
  python run_simulation.py --no-interactive
        """
    )
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to specific checkpoint (default: interactive selection)")
    parser.add_argument("--seed", type=int, default=None, 
                       help="Random seed for reproducibility (default: interactive selection)")
    parser.add_argument("--steps", type=int, default=None,
                       help="Number of simulation steps (default: from config or interactive)")
    parser.add_argument("--no-interactive", action='store_true',
                       help="Disable interactive mode (use defaults/args only)")
    args = parser.parse_args()
    
    run_simulation(
        checkpoint_path=args.checkpoint, 
        seed=args.seed, 
        max_steps=args.steps,
        interactive=not args.no_interactive
    )
