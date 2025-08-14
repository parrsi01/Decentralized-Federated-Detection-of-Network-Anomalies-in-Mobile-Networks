**Decentralized Federated Detection of Network Anomalies using MARL-XGBoost**


This repository contains the implementation of a fully decentralized anomaly detection framework that integrates Multi-Agent Reinforcement Learning (MARL) with XGBoost for scalable, robust, and efficient intrusion detection in network environments.

The system operates without a central server, enabling peer-to-peer model updates through trust-weighted aggregation and reinforcement-driven feature selection. Designed for non-IID data distributions and resource-constrained environments, it supports diverse network topologies (Ring, Star, Fully Connected, Random) and evaluates performance under challenging conditions such as agent dropout, packet loss, and adversarial peers.

Experiments are conducted on UNSW-NB15, CICIDS2017, and TON_IoT datasets, with comparisons against Hierarchical Decentralized Federated Learning (HDFL) and baseline methods.
