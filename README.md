# SpTRSV_Stuff

Sparse Triangular Solver (SpTRSV) is a major bottleneck in scientific computations. The inherently sequential nature of triangular solvers forces computations to proceed step by step, resulting in significant delays. However, for SpTRSV, the sparsity pattern of the matrix can provide opportunities for parallelism, which can be exploited to accelerate the process. In this study, I proposed a hardware architecture to speed up SpTRSV computations and analyzed its performance across different input matrix types. A detailed description is available in my [report](https://github.com/dilshansk/SpTRSV_Stuff/blob/main/documents/Evaluating%20SpTRSV%20Design%20Performance%20for%20FPGAs-Report.pdf) and [presentation](https://github.com/dilshansk/SpTRSV_Stuff/blob/main/documents/Evaluating%20SpTRSV%20Design%20Performance%20for%20FPGAs-Presentation.pdf).


## In a Nutshell

* The basic idea of a triangular solver is found under the terms of “forward substitution” and “backward substitution” in the literature. The process of solving a set of linear algebraic equations in the form of $Lx=y$, where $L$ is a lower triangular matrix is called forward substitution.

* The Directed Acyclic Graph can be used to visualize the dependencies of the rows of the triangle.

* We propose a single processing elements (PE) as shown below, which is equipped with all the computation units that are required to perform SpTRSV computation. Also, since we have multiple multiplication units inside the PE, we utilize the parallelism within the row (i.e., non-zero values inside the row) in our PE.

* We can further extend multiple PEs into processing element group (PEG) as follows to further distribute the parallelism within a row.

* We allocate different rows in a level to different PEGs to process in parallel (i.e., row parallelism). However, it is necessary to have every PEGs to be updated with all the newly solved xi values before moving to the next level. Since all to all broadcasting is very costly in FPGAs, we can use the ring communication method to broadcast data to all the PEGs as shown in the following figure.
