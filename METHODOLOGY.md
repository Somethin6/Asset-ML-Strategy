# Methodology note

The retained application follows a basic supervised-learning workflow over financial tabular data. It is useful as an early software/ML artifact, but its original design should not be read as a rigorous forecast study.

A stronger successor research design should use:

1. feature availability defined at each forecast origin;
2. rolling or expanding walk-forward evaluation;
3. persistence/drift and simple statistical baselines;
4. out-of-sample model selection boundaries;
5. explicit treatment of overlapping labels;
6. transaction-cost/slippage sensitivity for trading interpretations;
7. leakage and randomized-signal controls;
8. reporting of negative/null results alongside positive ones.

This document records the methodological standard for any future quantitative project derived from this codebase.
