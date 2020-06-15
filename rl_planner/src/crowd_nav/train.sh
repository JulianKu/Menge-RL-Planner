#!/bin/bash

python3 train.py $@ &
pid=$! # get pid of most recently launched process
freq_file=hz.txt

# while training not finished
while ps -p $pid > /dev/null
do
    # check hz rostopic
    python3 freq_writer.py /menge_sim_time -o $freq_file
    freq=$(tail -1 $freq_file)
    if [ $(echo "$freq < 1.0" | bc -l) -eq 1 ] # freq < thresh
    then
        # kill task
        kill $pid
        # restart task in background
        python3 train.py --restart $@ &
        pid=$!
    fi
done
