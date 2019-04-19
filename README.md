# Free Adversarial Training 
This repository belongs to the [Free Adversarial Training](http://arxiv.org/ "Free Adversarial Training") paper.
The implementation is inspired by [CIFAR10 Adversarial Example Challenge](https://github.com/MadryLab/cifar10_challenge "Madry's CIFAR10 Challenge") so to them we give the credit.


## Demo
To train a new robust model for free! run the following command specifying the replay parameter `m`:

```bash
python free_train.py -m 8
```

To evaluate a robust model using PGD-20 with 2 random restarts run:

```bash
python multi_restart_pgd_attack.py --model_dir $MODEL_DIR --num_restarts 2
```


## Requirements 
To install all the requirements plus tensorflow for multi-gpus run: (Inspired By [Illarion ikhlestov](https://github.com/ikhlestov/vision_networks "Densenet Implementation") ) 

```bash
pip install -r requirements/gpu.txt
```

Alternatively, to install the requirements plus tensorflow for cpu run: 
```bash
pip install -r requirements/cpu.txt
```

To prepare the data, please see [Datasets section](https://github.com/ashafahi/free_adv_train/tree/master/datasets "Dataset readme")
