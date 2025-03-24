
# Masked Visual-Tactile Pre-training for Robot Manipulation (ICRA24)

## Overview
This repository contains the official code for the research project "Masked Visual-Tactile Pre-training for Robot Manipulation," presented at ICRA24. The project focuses on enhancing robotic manipulation capabilities through a novel approach that integrates visual and tactile information.

### Key Links
- **Project Webpage**: [https://lqts.github.io/M2VTP/](https://lqts.github.io/M2VTP/)
- **Research Paper**: [IEEE](https://ieeexplore.ieee.org/document/10610933/) | [ResearchGate](https://www.researchgate.net/publication/378067504_Masked_Visual-Tactile_Pre-training_for_Robot_Manipulation)
- **Demo Video**: [Bilibili](https://www.bilibili.com/video/BV1pqkyYyEnp/?spm_id_from=333.1387.homepage.video_card.click)

## Features
This repository provides the following functionalities:
- **Environment Setup**: Instructions for configuring the necessary environment.
- **Pre-trained Model Integration**: Code for importing and utilizing pre-trained models.
- **Downstream Task Training**: Scripts for training models on specific manipulation tasks.
- **Model Evaluation**: Tools for testing trained models and visualizing training strategies.

## Getting Started

### Environment Setup
To set up the environment, please refer to the detailed instructions in the [Dependencies](https://github.com/LQTS/Pretraining_for_M2VTP?tab=readme-ov-file#dependencies) section.

### Importing Pre-trained Models
You can access the pre-trained model code [here](https://github.com/LQTS/Pretraining_for_M2VTP). Additionally, pre-trained models can be downloaded from [this link](https://1drv.ms/f/c/9054151f0ba654c9/EmdvA3bTnsNErIH6P5rsvlwBm-K3KRsRZKDm99xcYqoROA). Place the downloaded model and configuration files in the `model/vitac/model_and_config` directory. You can modify the directory information in `model/backbones/pre_model.py` as shown below:

```python
MODEL_REGISTRY = {
    "vt20t-reall-tmr05-bin-ft+dataset-BottleCap": {
        "config": "model/vitac/model_and_config/vt20t-reall-tmr05-bin-ft+dataset-BottleCap.json",
        "checkpoint": "model/vitac/model_and_config/vt20t-reall-tmr05-bin-ft+dataset-BottleCap.pt",
        "cls": VTT_ReAll,
    }
}
```

### Training a Policy
The repository includes a downstream task for bottle cap manipulation. To train the ShadowHand policy for this task, execute the following command:

```bash
python train_agent.py --task bottle_cap_vt --seed 123
```

**Note**: The training process requires at least two Nvidia 3090 GPUs—one for model training and the other for image rendering. Set the following environment variables accordingly:

```python
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ["MUJOCO_EGL_DEVICE_ID"] = "1"  # '1' is for image rendering, '0' is used for training.
```

### Testing a Policy
We provide a pre-trained policy that can be downloaded from [this link](https://1drv.ms/u/c/9054151f0ba654c9/EcRGCIisVj5EjPj4diXzFEsB25UKUT_ccNMlmlhV50Q-FA). You can also train your own model using the instructions above.

To perform testing, use the following command:

```bash
python eval_agent.py --task bottle_cap_vt --seed 123 --resume_model path/to/your/model.pt --test
```

The test results will be output to the console and saved to a specified file.

### Visualizing Policies
To save the operation process as a video, run the command:

```bash
python eval_agent.py --task bottle_cap_vt --seed 123 --resume_model path/to/your/model.pt --test --env_vis
```

The video will be saved in the `runs/videos` directory.
## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE/) file for details.


## Contact

If you have any questions or need support, please contact <a href="mailto:l_qingtao@zju.edu.cn"> Qingtao Liu</a> or <a href="mailto:qi.ye@zju.edu.cn">Qi Ye</a>.
.

## BibTeX
```
@inproceedings{liu2024m2vtp,
    title={Masked Visual-Tactile Pre-training for Robot Manipulation},
    author={Liu, Qingtao and Ye, Qi and Sun, Zhengnan and Cui, Yu and Li, Gaofeng and Chen, Jiming},
    booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
    year={2024},
    organization={IEEE}
} 
```