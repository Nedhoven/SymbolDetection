# SymbolDetection

The research during my Ph.D. studies at University of California, Irvine.

## Front End
The system will generate random sequence of complex symbols (constellations of BPSK, QPSK, 4QAM, 16QAM, and 64QAM).
There will be a precoder to encode the sequences in order to transmit the symbols

## Optimizer
After receiving the symbols, the optimizer categorizes the sequence in order to detect each individual symbol

## Back End
Using statistics methods (such as maximum likelihood and maximum a posteriori),
supervised machine learning techniques (logistic regression, nearest neighbor, decision trees, support vector machines, naive Bayes, and random forests),
and eventually, unsupervised machine learning techniques (clustering and k-means), the system can detect which symbol was originally transmitted.
