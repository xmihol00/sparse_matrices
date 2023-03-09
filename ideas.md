1. Develop a schema for storing sparse matrices, which is well suited e.g. for CPUs, GPUs or MCUs.
2. Try to match or improve upon the performance of SOTA implementation of dense matrix/vector multiplication, e.g on GPU:
    * cuBLAS dense 2048x2048x2048 matrix multiplication:
    ```
    GPU activities:   85.10%  49.055ms         1  49.055ms  49.055ms  49.055ms  sgemm_sm30_ldg_tex_tn_64x16x128x16x32
                      10.31%  5.9415ms         3  1.9805ms     992ns  2.9830ms  [CUDA memcpy HtoD]
                      4.59%  2.6455ms          1  2.6455ms  2.6455ms  2.6455ms  [CUDA memcpy DtoH]
    ``` 
    * naive dense 2048x2048x2048 matrix multiplication:
    ```
    GPU activities:   99.84%  4.88579s         1  4.88579s  4.88579s  4.88579s  dotRowsColumns
                      0.11%   5.2290ms         2  2.6145ms  2.5294ms  2.6997ms  [CUDA memcpy HtoD]
                      0.05%   2.6451ms         1  2.6451ms  2.6451ms  2.6451ms  [CUDA memcpy DtoH]
    ```
    * naive 2048x2048x2048 0.75 sparse matrix multiplication:
    ```
    GPU activities:   99.80%  3.00571s         1  3.00571s  3.00571s  3.00571s  dotRowsColumns
                      0.12%   3.5173ms         2  1.7587ms  949.20us  2.5681ms  [CUDA memcpy HtoD]
                      0.09%   2.6185ms         1  2.6185ms  2.6185ms  2.6185ms  [CUDA memcpy DtoH]
    ```
3. Possibly use RL to further improve performance by searching space of banchmarks.
4. Design specific HW units, which would implement specific instructions, that could further improve performance.
5. Implement a simulator, e.g. for CUDA assembly, for measuring execution including the new instructions.
6. Prove the concept and estimate possible performace gains of each new instruction.
