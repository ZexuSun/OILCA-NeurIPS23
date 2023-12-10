# Offline Imitation Learning with Variational Counterfactual Reasoning

This is the code for reproducing the results of the paper Offline Imitation Learning with Variational Counterfactual Reasoning accepted at NeurIPS'2023. The paper can be found [here](https://openreview.net/pdf?id=6d9Yxttb3w).

### Usage
Paper results were collected with [Deep Mind Control](https://github.com/google-deepmind/dm_control) (and [Causal World](https://sites.google.com/view/causal-world/home)). Networks are trained using [PyTorch 1.4.0](https://github.com/pytorch/pytorch) and Python 3.6.



You can also run OILCA on the setting used in [DWBC](https://arxiv.org/abs/2207.10050) by running `main_ivae_model.py` and `main_ivae_model.py`:

```
python main.py \
    --aux_dim="5" \  
    --epochs="30" \
    --task="cheetah_run" 
```

After the counterfactual model training, you can run the main.py to train the offline policy with the augmented data with the pretrained policy path:

```
python main_setting_demodice.py \
    --data_path="../dataset/dm_control_suite/" \  
    --env="cheetah_run" \
    --expert-policy-path="../learned_models/BC_all/bc_model_cheetah_run.pkl" 
```

### Bibtex
```
@inproceedings{sun2023offline,
  title     = {Offline Imitation Learning with Variational Counterfactual Reasoning},
  author    = {Sun, Zexu and He, Bowei and Liu, Jinxin and Chen, Xu and Ma, Chen and Zhang, Shuai},
  booktitle = {Proceedings of the 37th Conference on Neural Information Processing Systems},
  year      = {2023}
}
```

