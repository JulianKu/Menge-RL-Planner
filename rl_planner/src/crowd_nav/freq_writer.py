#!/usr/bin/env python3

import rospy as rp
from rostopic import ROSTopicHz, ROSTopicIOException, _check_master, get_topic_class
from argparse import ArgumentParser
from os.path import exists
from statistics import median


def main(topic, out):

    master_runnning = False
    while not master_runnning:
        try:
            _check_master()
            master_runnning = True
        except ROSTopicIOException:
            master_runnning = False
        rp.sleep(rp.Duration.from_sec(1))
    rp.init_node("frequency_writer", anonymous=True)
    rt = ROSTopicHz(-1)
    msg_class, real_topic, _ = get_topic_class(topic, blocking=True)  # pause hz until topic is published
    rp.Subscriber(real_topic, msg_class, rt.callback_hz, callback_args=topic)

    num_msgs = 0
    rates = []
    while num_msgs < 10 and not rp.is_shutdown():
        rp.rostime.wallsleep(1.0)
        ret = rt.get_hz(topic)
        if ret is not None:
            num_msgs += 1
            rates.append(ret[0])
    # remove outliers
    med_freq = median(rates)

    if exists(out):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(out, write_mode) as out_file:
        out_file.write("{:.1f}".format(med_freq) + "\n")

    return


if __name__ == '__main__':
    parser = ArgumentParser("return frequency [hz] of published topic")
    parser.add_argument("topic", type=str, help="name of the topic")
    parser.add_argument("--out", "-o", type=str, default="hz.txt", help="file to write frequencies to")
    args = parser.parse_args()
    main(args.topic, args.out)
