#!/bin/bash

function ensure_rosmaster {
    echo "making sure rosmaster is running"
    c=0
    master_pid=$(pgrep rosmaster)
    while [ -z $master_pid ]; do
        # give previous launched programs 5 sec to start rosmaster otherwise start self
        if [ $c == 5 ]; then
            echo "starting rosmaster"
            roscore &
        else
            sleep 1
        fi
        (( c++ ))
        master_pid=$(pgrep rosmaster)
    done
    echo "rosmaster running"
    return $master_pid
}

python3 train.py $@ &
train_pid=$! # get pid of most recently launched process (i.e. train.py)

master_pid=ensure_rosmaster

freq_file=hz.txt
python3 freq_writer.py /menge_sim_time -o $freq_file &
freq_writer_pid=$! # get pid of freq_writer.py)

# while training not finished
while ps -p $train_pid > /dev/null; do
    # check hz rostopic
    if [ -f "$freq_file" ]; then
        freq=$(tail -1 $freq_file)
        if [ $(echo "$freq < 2.0" | bc -l) -eq 1 ]; then # freq < thresh
            # kill task
            kill $train_pid
            wait $train_pid
            # restart task in background
            python3 train.py --restart $@ &
            train_pid=$!
            
            master_pid=ensure_rosmaster
            
            # remove last line from freq_file (freq < thresh) to avoid infinite restart
            head -n -1 $freq_file > temp.txt
            mv temp.txt $freq_file
        fi
    fi
done

kill $freq_writer_pid
kill $master_pid
