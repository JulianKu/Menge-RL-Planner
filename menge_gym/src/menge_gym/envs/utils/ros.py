from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
import subprocess
import threading
import queue
import psutil
from rospy import loginfo  # , logdebug, logerr
# from signal import SIGTERM, SIGKILL
from numpy import arctan2
from typing import Tuple, Dict


def pose2array(pose: Pose) -> Tuple[float, float, float]:
    """
    extract 2D position and orientation from ROS geometry_msgs/Pose message

    :param pose: ROS geometry_msgs/Pose message

    :return: list [x, y, phi] (2D position x,y + orientation phi)
    """
    # only 2D poses (position x,y + rotation around z)
    x = pose.position.x
    y = pose.position.y
    # orientation from quaternion
    q_z = pose.orientation.z
    q_w = pose.orientation.w
    phi = arctan2(2 * q_w * q_z, 1 - 2 * q_z * q_z)

    return x, y, phi


def obstacle2array(obs_pose: Pose) -> Tuple[float, float]:
    """
    extract 2D position from ROS geometry_msgs/Pose message

    :param obs_pose: ROS geometry_msgs/Pose message

    :return: list [x, y] (2D position x,y)
    """
    # only 2D point objects (x,y - no orientation)
    x = float(obs_pose.position.x)
    y = float(obs_pose.position.y)
    return x, y


def marker2array(marker: Marker) -> Tuple[float, float, float, float]:
    """
    extract 2D position, orientation and radius from ROS geometry_msgs/Pose message

    :param marker: ROS visualization_msgs/Marker message

    :return: list [x, y, q_w, q_z, r] (2D position x,y + orientation phi + radius r)
    """
    # only 2D poses (position x,y + rotation around z)
    x = marker.pose.position.x
    y = marker.pose.position.y
    # orientation quaternion
    q_z = marker.pose.orientation.z
    q_w = marker.pose.orientation.w
    phi = arctan2(2 * q_w * q_z,  1 - 2 * q_z * q_z)

    # radius
    r = marker.scale.x / 2
    return x, y, phi, r


def goal2msg(goal):
    """
    creates Marker message from goal
    :param goal: np.ndarray goal (x,y, radius)
    :return: Marker message
    """
    goal_msg = Marker()
    goal_msg.header.frame_id = "map"
    goal_msg.type = goal_msg.CYLINDER
    goal_msg.action = goal_msg.ADD
    goal = goal
    goal_msg.pose.position.x = goal[0]
    goal_msg.pose.position.y = goal[1]
    goal_msg.pose.orientation.w = 1.0
    goal_diameter = 2 * goal[2]
    goal_msg.scale.x = goal_diameter
    goal_msg.scale.y = goal_diameter
    goal_msg.scale.z = 0.2
    goal_msg.color.a = 0.7
    goal_msg.color.r = 1.0
    goal_msg.color.g = 0.9
    goal_msg.color.b = 0.0

    return goal_msg


def isProcessRunning(process_name: str) -> Tuple[bool, int]:
    """
    checks whether a process is already running

    :param process_name:
    :return:
    """
    proc_running = False
    pid = None
    for proc in psutil.process_iter():
        try:
            if process_name in proc.name():
                pid = proc.pid
                proc_running = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return proc_running, pid


# def kill_child_processes(parent_pid, sig=SIGTERM):
#     try:
#         parent = psutil.Process(parent_pid)
#         loginfo(parent)
#     except psutil.NoSuchProcess:
#         logerr("parent process not existing")
#         return
#     children = parent.children(recursive=True)
#     loginfo(children)
#     for process in children:
#         loginfo("try to kill child: " + str(process))
#         process.send_signal(sig)


def output_reader(proc: subprocess.Popen, out_queue: queue.Queue):
    for line in iter(proc.stdout.readline, b''):
        out_queue.put(line.decode('utf-8'))


class ROSHandle:

    def __init__(self):
        self.processes = {}  # type: Dict[int, subprocess.Popen]
        self.threads = {}  # type: Dict[int, threading.Thread]
        self.queue = queue.Queue()

        # check if "rosmaster" running already
        self.master_process = None
        self.master_running, self.master_PID = isProcessRunning("rosmaster")

        if not self.master_running:
            loginfo("No rosmaster running yet")
            loginfo("Start new rosmaster")
            self.master_process = subprocess.Popen(["roscore"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.master_PID = self.master_process.pid
            self.master_running = True
            self.log_output()

    def start_rosnode(self, pkg: str, executable: str, launch_cli_args: Dict[str, str] = None) -> int:
        """
        start a ROS node with arguments

        :param pkg: ROS package where node is specified
        :param executable: name of the node's executable
        :param launch_cli_args: dict of additional command line arguments {arg:value}
        """

        run_expr = ["rosrun", pkg, executable]

        if launch_cli_args:
            # resolve cli arguments
            for key in launch_cli_args:
                run_expr.append("-%s %s" % (key, launch_cli_args[key]))

        # run process
        proc = subprocess.Popen(run_expr, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pid = proc.pid
        self.processes[pid] = proc

        thr = threading.Thread(target=output_reader, args=(proc, self.queue))
        # thr.daemon = True
        thr.start()
        self.threads[pid] = thr

        self.log_output()

        return pid

    def log_output(self):

        out_queue = self.queue
        print('--- Print subprocess output ---')
        while True:
            try:
                print(out_queue.get_nowait(), end='')
            except queue.Empty:
                break
        print('-------------------------------')

    def terminate(self):
        """
        terminate all ros processes and the master itself
        """
        self.log_output()
        loginfo("Trying to kill all launched processes first")
        for pid in self.processes:
            process = self.processes[pid]
            # kill_child_processes(pid, SIGTERM)
            # process.kill()
            process.terminate()
            process.wait()
            thread = self.threads[pid]
            thread.join()
        self.processes = {}
        self.threads = {}
        with self.queue.mutex:
            self.queue.queue.clear()

        # if self.master_process:
        #     self.master_process.terminate()
        #     self.master_process.wait()
        #     self.master_process = None

    def terminateOne(self, pid):
        """
        terminate one ros process by its pid

        :param pid: process id for process to kill
        """
        self.log_output()
        loginfo("trying to kill process with pid: %s" % pid)
        proc = None
        try:
            proc = self.processes.pop(pid)
        except KeyError:
            try:
                proc = psutil.Process(pid)
            except psutil.NoSuchProcess:
                print("Process already terminated")
        # kill_child_processes(pid, SIGTERM)
        # proc.kill()
        if proc is not None:
            proc.terminate()
            proc.wait()

        try:
            thr = self.threads.pop(pid)
            thr.join()
        except KeyError:
            print("Thread already terminated")
