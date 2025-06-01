# Can Large Reasoning Models Self-Train?

This is the official PyTorch implementation of our paper ["Can Large Reasoning Models Self-Train?"](https://arxiv.org/abs/2505.21444) by [Sheikh Shafayat*](https://sheikhshafayat.github.io/), [Fahim Tajwar*](https://tajwarfahim.github.io/), [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/), [Jeff Schneider](https://www.cs.cmu.edu/~schneide/), and [Andrea Zanette](https://azanette.com/). Please see the [project website](https://self-rewarding-llm-training.github.io/) for more information about this work. For any questions/concerns related to the codebase, please reach out to [Fahim Tajwar](mailto:tajwarfahim932@gmail.com) and/or [Sheikh Shafayat](mailto:sheikhshafayat2@gmail.com).

## Citation

If you use this repo in your research, please consider citing our paper:

```
@misc{shafayat2025largereasoningmodelsselftrain,
      title={Can Large Reasoning Models Self-Train?}, 
      author={Sheikh Shafayat and Fahim Tajwar and Ruslan Salakhutdinov and Jeff Schneider and Andrea Zanette},
      year={2025},
      eprint={2505.21444},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.21444}, 
}
```

## Installation

In order for the installations to go smoothly, make sure you are operating from a GPU machine, typically one compatible with flash attention. It is ideal if you use the same GPU machines that you would use to run your experiments. This codebase is written on top of [verl](https://github.com/volcengine/verl), so you can also check out their very well written [installation guideline](https://verl.readthedocs.io/en/latest/start/install.html) if you need more information or if you have questions related to your particular machine setup.

We use Anaconda to manager our package installations in this project. If you do not have Anaconda setup in your machine, please check [this website](https://www.anaconda.com/docs/getting-started/anaconda/install) for more information. 

Now, please create a new conda environment with the correct dependencies (these may differ based on your compute resources, please update the packages accordingly). Run the following command first:

```
conda create -n online_rl python==3.10
conda activate online_rl
```

Next, insteall PyTorch. 

```
pip3 install torch torchvision torchaudio
```

If this does not work due to CUDA version mismatch on your device, please install the particular version of PyTorch that works for your machine from [here](https://pytorch.org/get-started/locally/).

Depending on your CUDA and PyTorch versions, you may or may not get to install FlashAttention directly via pip. To intsll it directly, do the following:

```
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
```

If this installation of FlashAttention seems buggy, you may have to install it from source. To do so, run the following:

```
pip install packaging
pip install ninja

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

Next, run the following commands in your shell:

```
pip install vllm
pip install wandb IPython matplotlib
pip install ray
```

Finally, you are ready to install this codebase. Make sure you are in an appropriate directory, then run the following:

```
cd srt
pip install -e .
```

Note that the packages required for this codebase are always evolving, and so may not work at the first try. Please contact the owners of individual packages in case something does not work. Best of luck!


## Running experiments

**GPU requirement**

We assume access to a single node consisting 8xH200 GPUs for these experiments. Please adjust the hyperparameters (e.g., batch size) accordingly if your GPUs have lower per GPU memory or other special requirements.

**Downloading datasets**

In order to use our codebase, you need to download and store the datasets in a certain format, using files in "examples/data_preprocess/<dataset_name>.py". We give a few example here so that you can get started:

To download the duplicated DAPO dataset, run the following command from an appropriate directory (make minor adjustments as you need):

```
python examples/data_preprocess/dapo.py --local_dir ~/data/dapo
```

To download the deduplicated DAPO dataset, but to remove all ground truth labels so that it can be used for SRT training (with labels produced by majority voting), please run the following command:

```
python examples/data_preprocess/dapo_with_label_noise.py --local_dir ~/data/dapo_unlabeled --label_noise 1.0 --add_self_consistency_labels
```

To download the compiled test dataset used in our paper, run the following command:

```
python examples/data_preprocess/srt_test_dataset.py --local_dir ~/data/srt_test_dataset
```

Finally, you can download the easy DAPO subset by running the following command:

```
# Labeled version
python examples/data_preprocess/dapo.py --local_dir ~/data/easy_dapo --dataset_path ftajwar/dapo_easy_one_third_sorted_by_frequency_of_majority_answer

# Unlabeled version
python examples/data_preprocess/dapo_with_label_noise.py --local_dir ~/data/easy_dapo_unlabeled --label_noise 1.0 --add_self_consistency_labels --dataset_path ftajwar/dapo_easy_one_third_sorted_by_frequency_of_majority_answer
```

**Running Reinforcement Learning with Ground Truth**

Once you have the datasets downloaded and preprocessed, you can run the following command to launch training with ground truth labels (we use DAPO and the RLOO algorithm for our example):

```
bash experiment_scripts/rl_with_ground_truth.sh
```

To run SRT on DAPO, with RLOO as the RL optimization algorithm, run the following command:

```
bash experiment_scripts/srt.sh
```

Please make appropriate changes to the scripts depending on your particular set of machines.


## Acknowledgements

This codebase is built on top of [verl](https://github.com/volcengine/verl), and we use the core functionalities in their codebase heavily. We thank the authors of verl for providing us with an extremely easy-to-work-with codebase!

Contemporary work such as [MM-UPT](https://github.com/waltonfuture/MM-UPT) have tried a similar idea for training multi-modal LLMs. We thank the authors for pointing it out, and would encourage interested users to look at their [codebase](https://github.com/waltonfuture/MM-UPT) and [paper](https://arxiv.org/abs/2505.22453).
