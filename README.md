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
Note that if you have trained a CIFAR-100 model, even for evaluation, you should pass the dataset argument. For examples:
```bash
python multi_restart_pgd_attack.py --model_dir $MODEL_DIR_TO_CIFAR100 --num_restarts 2 -d cifar100
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

If you find the paper or the code useful for your study, please consider citing the free training paper:
```bash
@ARTICLE{2019arXiv190412843S,
   author = {{Shafahi}, A. and {Najibi}, M. and {Ghiasi}, A. and {Xu}, Z. and 
	{Dickerson}, J. and {Studer}, C. and {Davis}, L. and {Taylor}, G. and {Goldstein}, T.},
    title = "{Adversarial Training for Free!}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1904.12843},
 primaryClass = "cs.LG",
 keywords = {Computer Science - Learning, Computer Science - Cryptography and Security, Computer Science - Computer Vision and Pattern Recognition, Statistics - Machine Learning},
     year = 2019,
    month = apr,
   adsurl = {http://adsabs.harvard.edu/abs/2019arXiv190412843S},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
