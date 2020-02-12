## PCA(ish) with [`ngraph`](https://github.com/NervanaSystems/ngraph/)

Computes the eigenvalues and eigenvectors of the covariance matrix of a centered matrix. 
So pretty much PCA, but without an interface. And only works with Eigen, so no vanilla fallback
with a plain Jacobi rotation implementation, or using MKL, GPUs..

I show how to add new kernels with `ngraph` (or at least with my limited understanding of it). I wrote a MeanOp and JacobiSVD kernel. The MeanOp should work but I ended up using an implementation of the mean just using operations already present in `ngraph`. You can probably do the same with JacobiSVD, but who has time for that?


## My conclusion:
 - `ngraph` has a good interface to add new operations, but you do need some fairly advanced C++ skills and probably a good IDE (recommend CLion) or be patient with gdb/lldb. Once you have the hang of it it's fairly straightforward.
 - I think eigendecomposition would be a good addition to `ngraph` (not sure if that is planned) as this is currently available with XLA. The `ngraph` community seems to be fairly active and responsive, so could ask them...
 - The reason I would choose `ngraph` over `XLA` is that it is quicker to build, as it does not depend on tensorflow code and it gives an option to use a prebuilt llvm/lib binary, rather than compile it yourself. However, XLA has more operations available, such as decompositions.

DISCLAIMER: I might have mixed up row and column major, but the computation I do in main.cpp is correct!