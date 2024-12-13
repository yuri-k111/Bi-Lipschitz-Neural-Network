This folder is based on the following github:

https://github.com/acfr/LBDN

It was modified for our purpose by adding and changing some features. We deleted files that were not used in our experiments to keep the folder concise and clear.

Some codes of https://github.com/ruigangwang7/StableNODE were also imported for the experiments.

Brief overview:

1. "env.yml" contains modules and their version to create a virtual environment where this code can be executed.

2. "main.py" was used for all the experiments around bi-Lipschitz control.

For example, run the following code to reproduce Table 1 (3) and Figure 2 (15)

for BLNN

  python main.py --mode train --model Toy --gamma 50 --layer legendre --scale small --dataset square_wave --epochs 100 -hd 64 -nhl 2 -convex 1

For other models, change the argument of --layer to Plain (for spectral normalization), Aol, Orthogon, SLL, Sandwich, bilipnet and change -hd accordingly. For LMN, see the folder  "partially_monotone".


To reproduce Figure 3 (and 16-21 in appendix), run

  python main.py --mode train --model Toy --gamma 50 --layer legendre --scale small --dataset linear1 --epochs 100 -hd 64 -nhl 2 -convex 1

Change --layer and -hd for other models. Also, changing linear1 (y=x) to linear50 will provide the target function y=50x. That way we obtain the experiment of Figure 1.

3. "opt_plot.py" was used for experiments around the algorithm of LFT.

For example, run the following code to reproduce Figures 7 and 8 in Appendix C.

  python opt_bl.py
