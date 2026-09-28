# Title

A Heckler in the Hidden State: Correctness Signals in Diffusion Language Models

# Authors

Angad Miglani, Samrath Singh Chadha, Kevin Li, Manas Venkata Sai Ravulapalli

# Abstract

Diffusion language models generate code by repeatedly updating a partially masked sequence. We ask whether their internal activations encode code correctness and whether that information can improve generation. Across six diffusion models, linear probes distinguish passing from failing attempts, with the strongest reads generally appearing beyond the early layers. Controls using small semantic mutations support a connection to correctness rather than surface style alone. In comparisons with model confidence, the probes offer no consistent advantage. Adding a probe-derived direction to the residual stream does not yield a dependable improvement in the tested steering settings, while the opposite direction degrades performance. The models carry information about correctness, but the tested interventions do not turn that information into better code.

# Comments

15 pages, 9 figures (4 main and 5 appendix), 4 tables. Includes supplementary code, results, and reproduction documentation.
