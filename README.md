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
    
3. Download the DistilBERT base model from Hugging Face in [hugging face](https://huggingface.co/distilbert-base-uncased) or in [distilbert-base-uncased](https://drive.google.com/drive/folders/1WFWyTFFOCEK0P5zvt2aQYX77XK9p9MYc?usp=sharing). Put "distilbert-base-uncased" under the directory of this repo.
    
## Data Preparation
Please refer to [`DATA.md`](DATA.md) for pre-training and downstream evaluation datasets.

## Pre-training
