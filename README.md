# Coalitional Bayesian Autoencoders for Explainable Unsupervised Deep Learning in Sensor Network

Python code for replicating the experiments and figures in the paper. Core dependencies are `agentMET4FOF` and `baetorch` for the agent-based system and Bayesian Autoencoder implementations. 

Investigated datasets are PRONOSTIA and ZEMA for condition monitoring.

## Contributions

- Formulation of two sensor attribution methods for BAE based on mean and epistemic uncertainty of negative log-likelihood estimate.
- Development of Coalitional BAE.
- Quantitative evaluation metrics for explainable HI based on covariate shift of sensors.
- Finding of misleading explanations due to correlation in outputs.

## Code descriptions

`unsupervised_zema_emc_bae_v5.py` : Main execution of agent network with customisable parameters. This produces raw csv results to be analysed.

`publication_cd_diagram.py` : Produces critical difference diagrams and record the aggregated table of results in csv. 

`publication_csv_latex.py`: Converts csv results from `publication_cd_diagram.py` into LATEX formatted content for tables.

`publication_explanation_quality.py` : Produces figure of examples for low (high) <img src="https://render.githubusercontent.com/render/math?math=G_{SDC}"> and <img src="https://render.githubusercontent.com/render/math?math=G_{SSER}">.

`publication_explanation_samples.py`: Samples of explanations comparing Centralised vs Coalition BAEs.

`publication_pearson.py` : Plots the complementary empirical cumulative distribution function (ECDF) showing the Pearson correlation between shifting and non-shifting sensors' attribution scores.

`publication_sample_corr`: Plots illustrative diagram of DNN prediction and its sensor attribution scores (Figure 1.)

## Figures

### Examples of Sensor Attribution Scores
(a) Equipment health indicator, (b) Sensor attribution scores
![example-sensor](./figures/example-sensor-attribution.png)

### Correlation between shifting and non-shifting sensors
Centralised BAE 
![Centralised-BAE-Sample](./figures/zema-bae-central-sample-8ea85cad5702975aaed0a3cc1112076121d83c.png)

Coalitional BAE 
![Coalitional-BAE-Sample](./figures/zema-bae-coalition-sample-a3ce2c815decfed70ad14f74666c82c1de7.png)

### Agent Network configuration
Centralised BAE 
![central-bae](./figures/crop_central.png)

Coalitional BAE 
![coalition-bae](./figures/crop_coalition.png)
