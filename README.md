#Privacy-Enhancing Sub-Sampling Meets Model Inversion Attacks

This repository contains the code and experiments from our bachelor's thesis at KTH, where we investigate how differentially private training combined with sub-sampling affects vulnerability to black-box model inversion attacks.

We reproduce and extend the attack proposed by He et al. and evaluate how optimizer choice (DP-SGD vs. Adam) and training strategies influence model performance and privacy leakage. All experiments were conducted in Google Colab using the CIFAR-10 dataset.

##Summary of Contributions
*Reproduced the black-box model inversion attack from He et al. (ACSAC 2019)
*Integrated Opacus to train image classifiers using DP-SGD with formal differential privacy guarantees
*Introduced sub-sampling as a privacy amplification technique
*Evaluated attack effectiveness using classification accuracy, PSNR, SSIM, and privacy margins
*Conducted all experiments in Google Colab with GPU acceleration

##Running the Experiments
This project is designed to run end-to-end in Google Colab. Each script corresponds to a step in the pipeline. GPU acceleration is recommended for all training and attack steps.

1. Train a baseline (non-private) model
`python training.py --dataset CIFAR10 --epochs 50` 

2. Train a private model using DP-SGD with sub-sampling
`python dp_training.py --epsilon 10 --delta 1e-5 --clip 3.0 --batch_size 256 --subsample 10000`

3. Run the black-box model inversion attack
`python inverse_blackbox.py --model_path path/to/model.pth --iterations 50` 

4. Evaluate privacy leakage and reconstruction quality
`python evaluate.py --metrics psnr ssim attack_accuracy`

This will generate:
*Classification and inversion accuracy
*PSNR and SSIM scores
*Reconstructed vs. original image comparisons

##Dependencies
To run locally, install the required packages:
`pip install torch torchvision opacus matplotlib scikit-learn numpy`
For Google Colab, these dependencies are pre-installed or easily added via pip.

##Dataset and Architecture
*Dataset: CIFAR-10 (using a subset of 7 classes)
*Model: Custom convolutional neural network (CIFAR10CNN)
*Training: 50 epochs per experiment
*Differential Privacy: Implemented using Opacus with RDP accounting

##Citation
If you use this work, please cite:

`@bachelorsthesis{abbas2025privacy,
  title={Privacy-Enhancing Sub-Sampling Meets Model Inversion Attacks},
  author={Abbas Mohammed, Matty Tomas },
  school={KTH Royal Institute of Technology},
  year={2025},
  note={Bachelor’s Thesis, TRITA-EECS-EX-2025:154}
}` 

Also consider citing the original attack paper:

`@inproceedings{he2019model,
  title={Model inversion attacks against collaborative inference},
  author={He, Zecheng and Zhang, Tianwei and Lee, Ruby B},
  booktitle={ACSAC},
  year={2019}
  ` 
}

##Acknowledgements
We thank our supervisor Leonhard Grosse for his guidance and support throughout this project
