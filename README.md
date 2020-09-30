# EMNLP-2020

This is the public repo for our EMNLP'20 paper: https://arxiv.org/abs/2009.07373.

You should be able to replicate the main results in the paper following the instructions below for TB-Dense data.
For I2b2 dataset, due to confidentiality and non-disclosure agreements, we are not able to publish anything relevant, but code sharing can be granted case by case upon data owner's agreement.

---

I included the environment that the code successfully ran on for your reference: `environment.yml`.

To replicate main results, simply run,
```bash
bash code/tbd/main_results.sh
```
---
To replicate results in the discussion (ablation study), run,
```bash
bash code/tbd/top{k}_results.sh
```
---
To conduct grid_search based on development data, run,
```bash
bash code/tbd/grid_search.sh
```