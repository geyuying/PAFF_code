import argparse
from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

import cv2
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

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
        log_dir = Path(__file__).parents[3] / "evaluation"
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
    args.log_dir = os.path.join(args.train_folder, 'test')
    log_dir = get_log_dir(args.log_dir)
    print('log_dir',log_dir)

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
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file)
    print(f"Best model: epoch {max(ranking, key=ranking.get)} with average sequences length of {max(ranking.values())}")

    for checkpoint, plan_dict in plan_dicts.items():
        epoch = checkpoint.stem.split("=")[1]

        ids, labels, plans, latent_goals = zip(
            *[
                (i, label, latent_goal, plan)
                for i, (label, plan_list) in enumerate(plan_dict.items())
                for latent_goal, plan in plan_list
            ]
        )
        latent_goals = torch.cat(latent_goals)
        plans = torch.cat(plans)
        np.savez(
            f"{log_dir / f'tsne_data_{epoch}.npz'}", ids=ids, labels=labels, plans=plans, latent_goals=latent_goals
        )


def evaluate_policy(model, env, lang_embeddings, args):
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_sequences = get_sequences(args.num_sequences)

    results = []
    plans = defaultdict(list)

    if not args.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    indx = 0
    for initial_state, eval_sequence in eval_sequences:
      #if indx >= 100:
      if indx in [122, 135, 165, 194, 23, 326, 356, 465, 497, 64, 128, 137, 182, 21, 319, 331, 405, 484, 60, 84]:
      #if indx in [0,20,21,23,33,35,38,63,64,81,93]:
        result, image_all = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings, val_annotations, args, plans
        )
        results.append(result)
        if True:
        #if result == 5:
            print('saving video', indx)
            video = cv2.VideoWriter("hulc/vis_hulc_before_256/"+str(indx)+ ".avi", cv2.VideoWriter_fourcc(*"XVID"), 15,
                                    (500, 500))
            for img in image_all:
                video.write(img)
            video.release()

        if not args.debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
      indx = indx + 1

    return results, plans


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, args, plans
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if args.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")

    image_all = []
    for subtask in eval_sequence:
        success, image_all = rollout(env, model, task_checker, args, subtask, lang_embeddings, val_annotations, plans, image_all)
        if success:
            success_counter += 1
        else:
            success_counter += 0
            #return success_counter, image_all
    return success_counter, image_all


def rollout(env, model, task_oracle, args, subtask, lang_embeddings, val_annotations, plans, imgs_all):
    if args.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65
    font_thickness = 1

    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    #print(lang_annotation)
    # get language goal embedding
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    model.reset()

    imgRGB = obs["rgb_obs"]["rgb_static"][0,0].cpu().numpy()
    imgRGB = np.transpose(imgRGB, (1,2,0))
    imgRGB = cv2.resize(
        ((imgRGB - imgRGB.min()) / (imgRGB.max() - imgRGB.min()) * 255).astype(np.uint8), (500, 500)
    )
    lang_goal = lang_annotation
    lang_textsize = cv2.getTextSize(lang_goal, font, font_scale, font_thickness)[0]
    lang_textX = (500 - lang_textsize[0]) // 2
    img = cv2.putText(imgRGB, lang_goal, org=(lang_textX, 480), fontFace=font, color=(0, 0, 0), fontScale=0.65,thickness=1, lineType=cv2.LINE_AA
    )[:, :, ::-1]
    # img = cv2.putText(
    #     imgRGB, f"{lang_annotation}", (100, 20), font, color=(0, 0, 0), fontScale=0.5, thickness=1
    # )[:, :, ::-1]
    imgs_all.append(img)

    start_info = env.get_info()

    plan, latent_goal = model.get_pp_plan_lang(obs, goal)
    plans[subtask].append((plan.cpu(), latent_goal.cpu()))

    for step in range(args.ep_len):
        action = model.step(obs, goal)
        obs, _, _, current_info = env.step(action)
        imgRGB = obs["rgb_obs"]["rgb_static"][0, 0].cpu().numpy()
        imgRGB = np.transpose(imgRGB, (1, 2, 0))
        imgRGB = cv2.resize(
            ((imgRGB - imgRGB.min()) / (imgRGB.max() - imgRGB.min()) * 255).astype(np.uint8), (500, 500)
        )
        lang_goal = lang_annotation
        lang_textsize = cv2.getTextSize(lang_goal, font, font_scale, font_thickness)[0]
        lang_textX = (500 - lang_textsize[0]) // 2
        img = cv2.putText(imgRGB, lang_goal, org=(lang_textX, 480), fontFace=font, color=(0, 0, 0), fontScale=0.65,
                          thickness=1, lineType=cv2.LINE_AA
                          )[:, :, ::-1]
        imgs_all.append(img)
        if args.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if args.debug:
                print(colored("success", "green"), end=" ")
            return True, imgs_all
    if args.debug:
        print(colored("fail", "red"), end=" ")
    return False, imgs_all


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

        checkpoints = checkpoints[:6]

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
