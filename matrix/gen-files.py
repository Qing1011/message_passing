#!/usr/bin/env python3


def gen_file(jn,fn):
    lines = [
        '#PBS -l walltime=23:00:00',
        '#PBS -l select=1:ncpus=1:mem=1gb',
        '#PBS -J 1-10',
        '',
        'SHARD_SIZE=10',
        'SHARD_ID=%d # from 0 to 9' % jn,
        '',
        'TASK_COUNT=1000',
        'TASK_ID=$((PBS_ARRAY_INDEX - 1))',
        '',
        'cd $HOME/gcc-wtm',
        'N=100000',
        'S=1000',
        '',
        'cd $HOME/gcc-wtm',
        '',
        'env \\',
        '    SHARD_SIZE=$SHARD_SIZE \\',
        '    SHARD_ID=$SHARD_ID \\',
        '    TASK_COUNT=$TASK_COUNT \\',
        '    TASK_ID=$TASK_ID \\',
        '    ./bin/gcc-wtm-pbs \\',
        '    ./data/z_list.txt \\',
        '    ./data/theta_list-small.txt \\',
        '    ./data/x_list-small.txt \\',
        '    $N $S',
    ]
    filename = 'job-%d-%d.pbs' % (fn,jn)
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def gen_files(job_n, file_n):
    for file_i in range(file_n):
        for job_i in range(job_n):
            print (job_i, file_i)
            gen_file(job_i,file_i)


#gen_files(10)
