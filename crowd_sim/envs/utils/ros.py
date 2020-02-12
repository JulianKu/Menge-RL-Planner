from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
import roslaunch


def pose2array(pose):
    """
    extract 2D position and orientation from ROS geometry_msgs/Pose message

    :param pose: ROS geometry_msgs/Pose message

    :return: list [x, y, q_w, q_z] (2D position x,y + 2D quaternion orientation q_w, q_z)
    """
    # only 2D poses (position x,y + rotation around z)
    x = pose.position.x
    y = pose.position.y
    # quaternion
    q_w = pose.orientation.w
    q_z = pose.orientation.z
    return [x, y, q_w, q_z]


def obstacle2array(obs_pose):
    """
    extract 2D position from ROS geometry_msgs/Pose message

    :param obs_pose: ROS geometry_msgs/Pose message

    :return: list [x, y] (2D position x,y)
    """
    # only 2D point objects (x,y - no orientation)
    x = obs_pose.position.x
    y = obs_pose.position.y
    return [x, y]


def marker2array(marker):
    """
    extract 2D position, orientation and radius from ROS geometry_msgs/Pose message

    :param marker: ROS visualization_msgs/Marker message

    :return: list [x, y, q_w, q_z, r] (2D position x,y + 2D quaternion orientation q_w, q_z + radius r)
    """
    # only 2D poses (position x,y + rotation around z)
    x = marker.pose.position.x
    y = marker.pose.position.y
    # quaternion
    q_w = marker.pose.orientation.w
    q_z = marker.pose.orientation.z
    # radius
    r = marker.scale.x
    return [x, y, q_w, q_z, r]

def start_roslaunch_file(pkg, launchfile, launch_cli_args=None):
    """
    start a ROS launchfile with arguments

    :param pkg: ROS package where launchfile is specified
    :param launchfile: name of the launch file
    :param launch_cli_args: dict of additional command line arguments {arg:value}

    :return: ROSLaunchParent of the launched file
    """

    # generate uuid for launch node
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    # make sure launchfile is provided with extension
    if not str(launchfile).endswith('.launch'):
        launchfile += '.launch'

    roslaunch_file = roslaunch.rlutil.resolve_launch_arguments([pkg, launchfile])
    if launch_cli_args is None:
        launch_expr = roslaunch_file
    else:
        # resolve cli arguments
        roslaunch_args = []
        for key in launch_cli_args:
            roslaunch_args.append(str(key) + ':=' + str(launch_cli_args[key]))

        launch_expr = [(roslaunch_file[0], roslaunch_args)]

    # launch file
    parent = roslaunch.parent.ROSLaunchParent(uuid, launch_expr)
    parent.start()

    return parent
