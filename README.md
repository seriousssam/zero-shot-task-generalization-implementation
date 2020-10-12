# Zero-shot Task Generalization: implementation with a new environment

## Status
The pdf for the project is here on github
The code for this project is presented -- it can definitely benefit from better documentation, but if you have any questions please email samrsabri@gmail.com 
Youtube video: https://youtu.be/gsufHS-bqxA.

## Intro
Project conducted with Chao Yu for Prof. Michael Yip's class Robot Reinforcement Learning (ECE 276C).

This project consists in:
1) A new simple maze-like game environment that can be used for reinforcement learning with task-generalization (loosely based on the environment described in Oh et al)
2) An implementation of Oh et al's paper: "Zero-Shot Task Generalization with Multi-Task Deep Reinforcement Learning" (https://arxiv.org/abs/1706.05064) -- which does not have published code.

## Files
- envandnets.py contains the env and the teacher training and network.
- dist.py contains the teachers data structure, the teacher testing, the policy distillation, and the GAE training, as well as the pskill and meta-controller networks.
- The majority of this code was written from scratch. For A3C, we used https://github.com/dgriff777/rl_a3c_pytorch.
