#!/usr/bin/env python3

import rospy as rp
from rostopic import ROSTopicHz, ROSTopicIOException, _check_master, get_topic_class
from argparse import ArgumentParser
from os.path import exists
from statistics import median
import sys


def main(topic, out):

    sub = None

    def shutdown():
        if sub is not None:
            sub.unregister()
        sys.exit(0)

    master_runnning = False
    while not master_runnning:
        try:
            _check_master()
            master_runnning = True
        except ROSTopicIOException:
            master_runnning = False
            rp.sleep(rp.Duration.from_sec(1))
    rp.on_shutdown(shutdown)
    rp.init_node("frequency_writer", anonymous=True)
    rt = ROSTopicHz(-1)
    msg_class, real_topic, _ = get_topic_class(topic, blocking=True)  # pause hz until topic is published
    sub = rp.Subscriber(real_topic, msg_class, rt.callback_hz, callback_args=topic)

    while not rp.is_shutdown():
        num_msgs = 0
        rates = []
        while num_msgs < 10:
            rp.rostime.wallsleep(1.0)
            ret = rt.get_hz(topic)
            if ret is not None:
                # only count actual freq returns
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
    parser = ArgumentParser(description="return frequency [hz] of published topic")
    parser.add_argument("topic", type=str, help="name of the topic")
    parser.add_argument("--out", "-o", type=str, default="hz.txt", help="file to write frequencies to")
    args = parser.parse_args()
    main(args.topic, args.out)
    sys.exit(0)
