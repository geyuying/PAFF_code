# Policy Adaptation from Foundation Model Feedback

[Paper](https://arxiv.org/abs/2212.07398) | [Project Page](https://geyuying.github.io/PAFF/)

![image](https://github.com/geyuying/PAFF_code/blob/main/imgs/PAFF.jpg?raw=true)

In this work, we propose **Policy Adaptation from Foundation model Feedback (PAFF)**. 
When deploying the trained policy to a new task or a new environment, we first let the policy **play** with randomly generated instructions to record the demonstrations. 
While the execution could be wrong, we can use the pre-trained foundation models to provide feedback by **relabeling** the demonstrations. 
This automatically provides new pairs of demonstration-instruction data for policy fine-tuning. 
We evaluate our method on a broad range of experiments with the focus on generalization on unseen objects, unseen tasks, unseen environments, and sim-to-real transfer. 

## Dependencies and Installation
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.7](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo

    ```bash
    git clone https://github.com/geyuying/PAFF_code
    cd PAFF_code
    ```

2. Install dependent packages

    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation
In this repo, we provide the training code for adapting the policy trained on [CALVIN](https://github.com/mees/calvin) Env A/B/C to Env D. 
Please refer to [HULC](https://github.com/lukashermann/hulc/tree/main) for downloading CALVIN Dataset.

## Training
**Stage-1**: Train your own policy on Env A/B/C of CALVIN dataset. In this repo, we follow [HULC](https://github.com/lukashermann/hulc/tree/main) for training the policy, but adopt the pre-trained [MDETR](https://github.com/ashkamath/mdetr) as
the visual and language encoder

**Stage-2**: Make the policy trained in the first stage ''play'' with a series of randomly generated language instructions in Env D of CALVIN dataset. We record these demonstrations including the visual observations and the robot’s actions by the trained policy.

```bash
cd Play
python hulc/evaluation/evaluate_policy_record.py  --dataset_path hulc/dataset/task_ABC_D --train_folder your_trained_policy_folder --last_k_checkpoints 1
```

**Stage-3**: Fine-tune CLIP for the ability to relabel the recorded demonstrations through reasoning about sequential visual observations with Spatio-Temporal Adapter ([ST-Adapter](https://arxiv.org/pdf/2206.13559)) on Env A/B/C of CALVIN dataset.\

```bash
cd CLIP_Finetune
python hulc/training.py datamodule.root_data_dir=hulc/dataset/task_ABC_D ~callbacks/rollout ~callbacks/rollout_lh
```

**Stage-4**: Use the fine-tuned CLIP in the third stage to relabel the recorded demonstrations through retrieving a language instruction among all possible language instructions. 

```bash
cd Relabel
python hulc/training.py datamodule.root_data_dir=hulc/record_D ~callbacks/rollout ~callbacks/rollout_lh
```

**Stage-5**: Fine-tuned the trained policy in the first stage on collected demonstration-instruction data.

```bash
cd Policy_Finetune
python hulc/training.py datamodule.root_data_dir=hulc/record_D_after_relabel  ~callbacks/rollout ~callbacks/rollout_lh
```

## Acknowledgement
Our code is based on the implementation of "What Matters in Language Conditioned Imitation Learning over Unstructured Data" <https://github.com/lukashermann/hulc>.

## Claim
In this repo, the training code has not been meticulously polished and organized :flushed:. Hope that it can provide you with some inspiration :sweat_smile:.


## Citation
If our code is helpful to your work, please cite:
```
@inproceedings{ge2023policy,
  title={Policy adaptation from foundation model feedback},
  author={Ge, Yuying and Macaluso, Annabella and Li, Li Erran and Luo, Ping and Wang, Xiaolong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19059--19069},
  year={2023}
}
```
