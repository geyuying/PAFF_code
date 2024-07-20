from itertools import chain
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import os

from hulc.datasets.base_dataset import BaseDataset
from hulc.datasets.utils.episode_utils import lookup_naming_pattern

logger = logging.getLogger(__name__)


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: str) -> Dict[str, np.ndarray]:
    return np.load(filename)


class DiskDataset(BaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        skip_frames: int = 0,
        save_format: str = "npz",
        pretrain: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        print('self.pretrain', self.pretrain)
        self.skip_frames = skip_frames

        if self.with_lang:
            self.episode_lookup, self.episode_lookup_seq, self.episode_lookup_task, self.lang_lookup, self.lang_ann, self.lang_raw = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_datasets_dir, self.save_format)


    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.

        Args:
            file_idx: index of starting frame.

        Returns:
            Path to file.
        """
        return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]
        seq_id = self.episode_lookup_seq[idx]
        task_id = self.episode_lookup_task[idx]
        end_idx = start_idx + window_size
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        #keys.append("scene_obs")
        episodes = [self.load_file(
            str(self.abs_datasets_dir) + '/sequence_' + str(seq_id).zfill(5) + '_' + str(task_id) + '_' + str(
                file_idx).zfill(4) + '.npz') for file_idx in range(start_idx, end_idx)]

        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
            episode["raw"] = self.lang_raw[self.lang_lookup[idx]]
        return episode

    def _build_file_indices_lang(self, abs_datasets_dir: Path) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        episode_lookup_seq = []
        episode_lookup_task = []

        try:
            print("trying to load lang data from: ", abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        except Exception:
            print("Exception, trying to load lang data from: ", abs_datasets_dir / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["emb"]  # length total number of annotations
        lang_raw = lang_data["language"]["ann"]
        lang_lookup = []
        for i, record_idx in enumerate(ep_start_end_ids):

            seq_id = int(record_idx / 40)
            task_id = int((record_idx % 40) / 8)
            step_id = int((record_idx % 40) % 8)

            start_idx = step_id * 12
            end_idx = start_idx + 32

            if self.pretrain:
                start_idx = max(start_idx, end_idx + 1 - self.min_window_size - self.aux_lang_loss_window)
            assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx - self.min_window_size+1):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                    episode_lookup_seq.append(seq_id)
                    episode_lookup_task.append(task_id)

                cnt += 1

        return np.array(episode_lookup), np.array(episode_lookup_seq), np.array(episode_lookup_task), lang_lookup, lang_ann, lang_raw

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)
