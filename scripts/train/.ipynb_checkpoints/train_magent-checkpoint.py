import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

sys.path.append("../../")
from HSPARC.config import get_config
from HSPARC.envs.magent.adv_pursuit_wrappers import AdvPursuit_w_PretrainedOpp
from HSPARC.envs.magent.battle_wrappers import Battle_w_PretrainedOpp
from HSPARC.runner.shared.magent_runner import MAgentRunner as Runner
from HSPARC.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


"""Train script for MAgent environments."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "magent":
                env_args = {
                    "map_name": all_args.scenario,
                    "map_size": all_args.map_size,
                    "max_cycles": all_args.episode_length,
                    "minimap_mode": all_args.minimap_mode,
                    "pretrained_ckpt": all_args.pretrained_ckpt,
                    "global_reward": all_args.global_reward,
                    "seed": all_args.seed + rank * 1000
                }
                
                if all_args.scenario == "adversarial_pursuit":
                    env = AdvPursuit_w_PretrainedOpp(env_args=env_args)
                elif all_args.scenario == "battle":
                    env = Battle_w_PretrainedOpp(env_args=env_args)
                else:
                    raise NotImplementedError(f"Scenario {all_args.scenario} not supported")
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "magent":
                env_args = {
                    "map_name": all_args.scenario,
                    "map_size": all_args.map_size,
                    "max_cycles": all_args.episode_length,
                    "minimap_mode": all_args.minimap_mode,
                    "pretrained_ckpt": all_args.pretrained_ckpt,
                    "global_reward": all_args.global_reward,
                    "seed": all_args.seed * 50000 + rank * 10000
                }
                
                if all_args.scenario == "adversarial_pursuit":
                    env = AdvPursuit_w_PretrainedOpp(env_args=env_args)
                elif all_args.scenario == "battle":
                    env = Battle_w_PretrainedOpp(env_args=env_args)
                else:
                    raise NotImplementedError(f"Scenario {all_args.scenario} not supported")
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            
            return env

        return init_env

    if all_args.eval_episodes == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.eval_episodes)])


def parse_args(args, parser):
    # MAgent specific arguments
    parser.add_argument('--scenario', type=str, default='adversarial_pursuit', 
                       choices=['adversarial_pursuit', 'battle'])
    parser.add_argument('--map_size', type=int, default=45,
                       help='Size of the map (45, 50, 55, 60, 65, 70 for pursuit; 25-100 for battle)')
    parser.add_argument('--n_agent', type=int, default=25,
                       help='Number of controlled agents (red team)')
    
    # Environment configuration
    parser.add_argument('--minimap_mode', action='store_true', default=True,
                       help='Use minimap mode for observations')
    parser.add_argument('--global_reward', action='store_true', default=False,
                       help='Use global reward instead of individual rewards')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                       help='Path to pretrained opponent model')
    
    # MAgent specific training parameters
    parser.add_argument('--use_agent_specific_state', action='store_true', default=False,
                       help='Use agent-specific state representation')
    parser.add_argument('--use_team_communication', action='store_true', default=False,
                       help='Enable team communication mechanisms')
    parser.add_argument('--add_position_state', action='store_true', default=True,
                       help='Add position information to state')
    parser.add_argument('--add_health_state', action='store_true', default=True,
                       help='Add health information to state')

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("MAgent config: ", all_args)
    print(f"Training {all_args.scenario} with map size {all_args.map_size}, {all_args.n_agent} agents")

    # Algorithm configuration
    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "RW_comm" or all_args.algorithm_name == "RW_comm_dec":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        # For HSPARC algorithm
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False

    if "dec" in all_args.algorithm_name:
        all_args.dec_actor = True
        all_args.share_actor = False

    # CUDA configuration
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Directory setup
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario / f"map{all_args.map_size}" / all_args.algorithm_name / all_args.experiment_name
    
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # WandB configuration
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.scenario) + "_" +
                              str(all_args.map_size) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=f"{all_args.scenario}_map{all_args.map_size}",
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        import time
        timestr = time.strftime("%y%m%d-%H%M%S")
        curr_run = all_args.prefix_name + "-" + timestr
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # Process title
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + 
        str(all_args.scenario) + "-" + str(all_args.experiment_name) + "@" + 
        str(all_args.user_name))

    # Seed setup
    torch.manual_seed(all_args.alg_seed)
    torch.cuda.manual_seed_all(all_args.alg_seed)
    np.random.seed(all_args.alg_seed)

    # Environment setup
    print("Creating training environment...")
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    
    # Get number of agents from environment
    num_agents = envs.n_agents
    print(f"Environment created with {num_agents} agents")
    
    # Validate agent count
    if all_args.scenario == "adversarial_pursuit":
        expected_agents = all_args.n_agent
        if num_agents != expected_agents:
            print(f"Warning: Expected {expected_agents} agents but environment has {num_agents}")
    elif all_args.scenario == "battle":
        expected_agents = all_args.n_agent
        if num_agents != expected_agents:
            print(f"Warning: Expected {expected_agents} agents but environment has {num_agents}")

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # Create and run runner
    runner = Runner(config)
    runner.run()

    # Cleanup
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])