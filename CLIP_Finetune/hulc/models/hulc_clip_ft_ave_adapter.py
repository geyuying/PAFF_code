import logging
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import yaml
import hydra
import numpy as np
import random
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
import os
from PIL import Image
import matplotlib.pyplot as plt
from hulc.models.decoders.action_decoder import ActionDecoder
from hulc.utils.distributions import State
from hulc.models.encoders.clip_lang_encoder import LangClip
from hulc.models.encoders.clip_vision_encoder_ft import VisionClip

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class Hulc(pl.LightningModule):
    """
    The lightning module used for training.

    Args:
        perceptual_encoder: DictConfig for perceptual_encoder.
        plan_proposal: DictConfig for plan_proposal network.
        plan_recognition: DictConfig for plan_recognition network.
        language_encoder: DictConfig for language_encoder.
        language_goal: DictConfig for language_goal encoder.
        visual_goal: DictConfig for visual_goal encoder.
        action_decoder: DictConfig for action_decoder.
        kl_beta: Weight for KL loss term.
        kl_balancing_mix: Weight for KL balancing (as in https://arxiv.org/pdf/2010.02193.pdf).
        state_recons: If True, use state reconstruction auxiliary loss.
        state_recon_beta: Weight for state reconstruction loss term.
        lang_recons: If True, use BC-Z language regression auxiliary loss.
        lang_recon_beta: Weight for language reconstruction loss term.
        lang_contrastive: If True, use MIA cross-modality matching auxiliary loss.
        lang_contrastive_beta: Weight for cross-modality matching loss term.
        optimizer: DictConfig for optimizer.
        lr_scheduler: DictConfig for learning rate scheduler.
        distribution: DictConfig for plan distribution (continuous or discrete).
        val_instructions: DictConfig with validation language instructions for each task.
        img_lang_matching_clip: If True, use CLIP contrastive auxiliary loss.
        lang_clip_beta: Weight for CLIP contrastive loss.
        replan_freq: After how many steps generate new plan (only for inference).
        lang_decoder: DictConfig for language regression network for BC-Z language regression loss.
        lang_discriminator: DictConfig for discriminator network for MIA cross-modality matching loss.
        clip_proj: DictConfig for projection network for CLIP contrastive loss.
    """

    def __init__(
        self,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        img_lang_matching_clip: bool,
        lang_clip_beta: float,
    ):
        super(Hulc, self).__init__()
        self.img_lang_matching_clip = img_lang_matching_clip
        self.lang_clip_beta = lang_clip_beta
        if img_lang_matching_clip:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.clip_text_encoder = LangClip()
        self.clip_vision_encoder = VisionClip()

        self.modality_scope = "lang"
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler

        self.save_hyperparameters()

        ann_file = 'hulc/dataset/task_ABC_D/training/mdetr/auto_lang_ann.npy'
        ann_content = np.load(ann_file, allow_pickle=True).item()
        ann_all = ann_content['language']['ann']
        task_all = ann_content['language']['task']
        ann_dict = {}
        task_dict = {}
        task_id = 0

        for i in range(len(ann_all)):
            ann = ann_all[i]
            ann = "A white robot arm " + ann
            task = task_all[i]
            if task not in ann_dict.keys():
                ann_dict[task] = [ann]
                task_dict[task_id] = task
                task_id = task_id + 1
            else:
                ann_cur = ann_dict[task]
                if ann not in ann_cur:
                    ann_cur.append(ann)
                    ann_dict[task] = ann_cur

        ann_embeddings = None
        for key in ann_dict.keys():
            ann_key = ann_dict[key]
            ann_key_embeddings = self.clip_text_encoder(ann_key)
            print(key, ann_key_embeddings.shape)
            ann_key_embeddings = torch.mean(ann_key_embeddings, 0).unsqueeze(0).cpu().numpy()
            if ann_embeddings is None:
                ann_embeddings = ann_key_embeddings
            else:
                ann_embeddings = np.concatenate((ann_embeddings, ann_key_embeddings), 0)

        self.ann_embeddings = torch.Tensor(ann_embeddings)
        print('ann_embeddings:', self.ann_embeddings.shape)
        self.task_dict = task_dict

    @property
    def num_training_steps(self) -> int:
        """
        Total training steps inferred from datamodule and devices.

        Returns:
            Number of estimated training steps.
        """
        assert isinstance(self.trainer, pl.Trainer)
        combined_loader_dict = self.trainer.datamodule.train_dataloader()  # type: ignore
        dataset_lengths = [len(combined_loader_dict[k]) for k in combined_loader_dict.keys()]
        dataset_size = max(dataset_lengths)
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices  # type: ignore
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs  # type: ignore

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:  # type: ignore
            return self.trainer.max_steps  # type: ignore
        return max_estimated_steps

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        """
        Set up warmup steps for learning rate scheduler.

        Args:
            num_training_steps: Number of training steps, if < 0 infer from class attribute.
            num_warmup_steps: Either as absolute number of steps or as percentage of training steps.

        Returns:
            num_training_steps: Number of training steps for learning rate scheduler.
            num_warmup_steps: Number of warmup steps for learning rate scheduler.
        """
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        num_warmup_steps = int(num_warmup_steps)
        return num_training_steps, num_warmup_steps

    def configure_optimizers(self):
        params_adapter = []
        for name, p in self.named_parameters():
            if 'down_projs' in name or "up_projs" in name or "depth_convs" in name:
                print(name)
                params_adapter.append(p)

        optimizer = hydra.utils.instantiate(self.optimizer_config, params=params_adapter)
        if "num_warmup_steps" in self.lr_scheduler:
            self.lr_scheduler.num_training_steps, self.lr_scheduler.num_warmup_steps = self.compute_warmup(
                num_training_steps=self.lr_scheduler.num_training_steps,
                num_warmup_steps=self.lr_scheduler.num_warmup_steps,
            )
            rank_zero_info(f"Inferring number of training steps, set to {self.lr_scheduler.num_training_steps}")
            rank_zero_info(f"Inferring number of warmup steps from ratio, set to {self.lr_scheduler.num_warmup_steps}")
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.


        Returns:
            loss tensor
        """
        lang_clip_loss, total_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        encoders_dict = {}
        batch_size: Dict[str, int] = {}
        total_bs = 0
        for self.modality_scope, dataset_batch in batch.items():
            visual_input = dataset_batch["rgb_obs"]["rgb_static"]
            visual_features = self.clip_vision_encoder(visual_input)

            lang_features = self.ann_embeddings.to(self.device)
            lang_labels = dataset_batch["raw_label"].reshape(-1)

            logits_per_image, loss = self.clip_loss(visual_features, lang_features, lang_labels)
            lang_clip_loss += loss
            batch_size[self.modality_scope] = dataset_batch["actions"].shape[0]
            total_bs += dataset_batch["actions"].shape[0]


        if self.img_lang_matching_clip:
            total_loss = total_loss + self.lang_clip_beta * lang_clip_loss
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        return total_loss

    def clip_loss(self, image_features, lang_features, labels):
        """
        CLIP style contrastive loss, adapted from 'Learning transferable visual models from natural language
        supervision' by Radford et al.
        We maximize the cosine similarity between the visual features of the sequence i and the corresponding language
        features while, at the same time, minimizing the cosine similarity between the current visual features and other
        language instructions in the same batch.

        Args:
            seq_vis_feat: Visual embedding.
            encoded_lang: Language goal embedding.
            use_for_aux_loss: Mask of which sequences in the batch to consider for auxiliary loss.

        Returns:
            Contrastive loss.
        """
        assert self.img_lang_matching_clip is not None

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        loss = cross_entropy(logits_per_image, labels)
        return logits_per_image, loss


    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal networks, as well as the
            episode indices.
        """
        output = {}
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        for self.modality_scope, dataset_batch in batch.items():
            visual_input = dataset_batch["rgb_obs"]["rgb_static"]
            visual_features = self.clip_vision_encoder(visual_input)

            lang_features = self.ann_embeddings.to(self.device)
            lang_labels = dataset_batch["raw_label"].reshape(-1)

            logits_per_image, val_pred_clip_loss = self.clip_loss(visual_features, lang_features, lang_labels)
            self.log("val/val_pred_clip_loss", val_pred_clip_loss, sync_dist=True)

            sim = logits_per_image.detach().cpu().numpy()
            pred_idx = np.argmax(sim, -1).reshape(-1)
            labels = lang_labels.cpu().numpy()
            correct = (pred_idx == labels).sum()
            self.correct_all = self.correct_all + correct
            self.num_all = self.num_all + len(labels)
            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]

            bz = visual_input.shape[0]
            for j in range(bz):
                ann_index = sim.argmax(-1)[j]
                label = lang_labels.cpu().numpy()[j]
                score = sim[j][ann_index]
                if score < 4.0 or label == ann_index:
                    continue
                task = self.task_dict[ann_index]
                gt_task = self.task_dict[label]
                print(score, task, gt_task)
        return output

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0


    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        self.correct_all = 0.0
        self.num_all = 0.0
        logger.info(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        result = self.correct_all / self.num_all
        print('Top-1:', result, self.correct_all, self.num_all)
        logger.info(f"Finished validation epoch {self.current_epoch}")

