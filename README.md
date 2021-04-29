# Coalitional Bayesian Autoencoders for Explainable Unsupervised Deep Learning in Sensor Network

Python code for replicating the experiments and figures in the paper. Core dependencies are `agentMET4FOF` and `baetorch` for the agent-based system and Bayesian Autoencoder implementations. 

## Code descriptions

`unsupervised_zema_emc_bae_v5.py` : Main execution of agent network with customisable parameters. This produces raw csv results to be analysed.
`publication_cd_diagram.py` : Produces critical difference diagrams and record the aggregated table of results in csv. 
`publication_csv_latex.py`: Converts csv results from `publication_cd_diagram.py` into LATEX formatted content for tables.
`publication_explanation_quality.py` : Produces figure of examples for low (high) $G_{SDC}$ and $G_{SSER}$.
`publication_explanation_samples.py`: Samples of explanations comparing Centralised vs Coalition BAEs.
`publication_pearson.py` : Plots the complementary empirical cumulative distribution function (ECDF) showing the Pearson correlation between shifting and non-shifting sensors' attribution scores.
`publication_sample_corr`: Plots illustrative diagram of DNN prediction and its sensor attribution scores (Figure 1.)

## Figures




