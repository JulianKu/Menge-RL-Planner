#!/bin/bash

function ensure_rosmaster {
    echo "making sure rosmaster is running"
    c=0
    master_pid=$(pgrep rosmaster)
    while [ -z $master_pid ]
    do
        # give previous launched programs 5 sec to start rosmaster otherwise start self
        if [ $c == 5 ]
        then
            echo "starting rosmaster"
            roscore &
        else
            sleep 1
        fi
        (( c++ ))
        master_pid=$(pgrep rosmaster)
    done
    echo "rosmaster running"
}

python3 train.py $@ &
pid=$! # get pid of most recently launched process
freq_file=hz.txt

ensure_rosmaster

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
        wait $pid
        # restart task in background
        python3 train.py --restart $@ &
        pid=$!
        
        ensure_rosmaster
    fi
done
