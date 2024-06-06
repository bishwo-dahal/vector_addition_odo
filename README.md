# THIS IS A COPY OF REPOSITORY OF ORNL OLCF WHICH CAN BE FOUND AT https://code.ornl.gov/olcf/

# vector_addition

This repository contains vector addition codes written for multiple parallel programming models as well as combinations of them. Each directory contains a stand-alone code, and the name of the directories should indicate the programming model(s) used. 

## Helpful Tips

* If you want to convince yourself that a particular code is actually running on a GPU, you can include `srun ... rocprof --stats <exe>` and check to see if the resulting results.stats.csv file has details about a kernel function. If this file was not written, it might also indicate the code didn't run on a GPU.

    * For MPI versions of the code you might want to use `rocprof -o results_${SLURM_PROCID}.csv --stats` so that each MPI ranks writes out its own .csv file.

    * If you're only running on a single node with multiple MPI ranks, and you'd like all ranks to write to the same file in a readable format (i.e., not garbled), you can use `srun ... bash -c 'rocprof -o results.${SLURM_PROCID}.csv <exe>'`.

## Reporting Issues

If you find any problems running these codes, please feel free to open a GitLab Issue.
