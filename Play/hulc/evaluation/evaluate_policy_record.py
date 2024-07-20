import argparse
from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
import cv2

from calvin_env.envs.play_table_env import get_env
from hulc.evaluation.multistep_sequences import get_sequences
from hulc.evaluation.utils import get_default_model_and_env, get_env_state_for_initial_condition, join_vis_lang
from hulc.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint

logger = logging.getLogger(__name__)


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path('./results')
        #log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")
    print(f"logging to {log_dir}")
    return log_dir


class CustomModel:
    def __init__(self):
        logger.warning("Please implement these methods as an interface to your custom model architecture.")
        raise NotImplementedError

    def reset(self):
        """
        This is called
        """
        raise NotImplementedError

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        raise NotImplementedError


class CustomLangEmbeddings:
    def __init__(self):
        logger.warning("Please implement these methods in order to use your own language embeddings")
        raise NotImplementedError

    def get_lang_goal(self, task_annotation):
        """
        Args:
             task_annotation: langauge annotation
        Returns:

        """
        raise NotImplementedError


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(total_results, plan_dicts, args):
    log_dir = get_log_dir(args.log_dir)

    sequences = get_sequences(args.num_sequences)

    current_data = {}
    ranking = {}
    for checkpoint, results in total_results.items():
        epoch = checkpoint.stem.split("=")[1]
        print(f"Results for Epoch {epoch}:")
        avg_seq_len = np.mean(results)
        ranking[epoch] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

        current_data[epoch] = data

        print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}



def evaluate_policy(model, env, lang_embeddings, args):
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    #val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    ann_file = 'hulc/dataset/task_ABC_D/training/mdetr/auto_lang_ann.npy'
    ann_content = np.load(ann_file, allow_pickle=True).item()
    ann_all = ann_content['language']['ann']
    task_all = ann_content['language']['task']
    embed_all = ann_content['language']['emb']
    ann_dict = {}
    ann_read = []
    for i in range(len(ann_all)):
        ann = ann_all[i]
        task = task_all[i]
        emb = embed_all[i]

        if task not in ann_dict.keys():
            ann_dict[task] = emb
            ann_read.append(ann)
        else:
            if ann not in ann_read:
                ann_read.append(ann)
                emb_cur = ann_dict[task]
                emb_cur = np.concatenate((emb_cur, emb), 0)
                ann_dict[task] = emb_cur

    eval_sequences = get_sequences(args.num_sequences)

    results = []
    plans = defaultdict(list)

    if not args.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    seq_num = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, ann_dict, args, plans, seq_num
        )
        seq_num = seq_num + 1
        results.append(result)
        if not args.debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
    return results, plans


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, ann_dict, args, plans, seq_num
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    if (seq_num % 50) == 0:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    else:
        env.reset_robot(robot_obs=robot_obs)

    success_counter = 0
    task_num = 0
    for subtask in eval_sequence:
        env.reset_robot(robot_obs=robot_obs)
        f = open('hulc/record_given.txt','a')
        f.write(str(seq_num))
        f.write(' ')
        f.write(str(task_num))
        f.write(' ')
        f.write(subtask)
        f.write('\n')
        #success = rollout(env, model, task_checker, args, subtask, ann_dict, plans, seq_num, task_num)
        task_num = task_num + 1
    return success_counter


def rollout(env, model, task_oracle, args, subtask, ann_dict, plans, seq_num, task_num):
    obs = env.get_obs()
    # get lang annotation for subtask
    ann_all = ann_dict[subtask]
    lang_ann = ann_all[np.random.randint(ann_all.shape[0])]
    # get language goal embedding
    goal = {"lang": torch.from_numpy(lang_ann).cuda().unsqueeze(0).float()}
    model.reset()
    start_info = env.get_info()

    plan, latent_goal = model.get_pp_plan_lang(obs, goal)
    plans[subtask].append((plan.cpu(), latent_goal.cpu()))


    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    mean_static = np.tile(np.reshape(mean, (-1, 1, 1)), (1, 200, 200))
    std_static = np.tile(np.reshape(std, (-1, 1, 1)), (1, 200, 200))
    mean_gripper = np.tile(np.reshape(mean, (-1, 1, 1)), (1, 84, 84))
    std_gripper = np.tile(np.reshape(std, (-1, 1, 1)), (1, 84, 84))

    save_dir = 'hulc/record_D'

    for step in range(args.ep_len):
        action = model.step(obs, goal)
        rgb_static = obs["rgb_obs"]["rgb_static"][0, 0].cpu().numpy() * std_static + mean_static
        rgb_static = rgb_static * 255
        rgb_gripper = obs["rgb_obs"]["rgb_gripper"][0, 0].cpu().numpy() * std_gripper + mean_gripper
        rgb_gripper = rgb_gripper * 255
        rgb_static = np.transpose(rgb_static, (1,2,0)).astype(np.uint8)
        rgb_gripper = np.transpose(rgb_gripper, (1,2,0)).astype(np.uint8)
        actions_save = action.cpu().numpy().reshape(-1)
        robot_obs = obs['robot_obs_raw'].cpu().numpy().reshape(-1)

        save_name = save_dir + 'sequence_' + str(seq_num).zfill(5) + '_' + str(task_num) + '_' + str(step).zfill(4) + '.npz'
        print(save_name)
        np.savez(save_name, rel_actions=actions_save, rgb_static=rgb_static, rgb_gripper=rgb_gripper,
                robot_obs=robot_obs)
        #break
        obs, _, _, current_info = env.step(action)

    return False


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )
    parser.add_argument("--custom_lang_embeddings", action="store_true", help="Use custom language embeddings.")

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # Do not change
    args.ep_len = 256
    args.num_sequences = 500

    lang_embeddings = None
    if args.custom_lang_embeddings:
        lang_embeddings = CustomLangEmbeddings()

    # evaluate a custom model
    if args.custom_model:
        model = CustomModel()
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, lang_embeddings, args)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        results = {}
        plans = {}
        for checkpoint in checkpoints:
            model, env, _, lang_embeddings = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                lang_embeddings=lang_embeddings,
                device_id=args.device,
            )
            results[checkpoint], plans[checkpoint] = evaluate_policy(model, env, lang_embeddings, args)

        print_and_save(results, plans, args)


if __name__ == "__main__":
    main()
