import math
from typing import List, Optional

import rospy
import tf.transformations
import tf2_ros
from geometry_msgs.msg import TransformStamped
from ros_osv.py_openshowvar import OpenShowVar
from std_msgs.msg import Bool

DO = 25
FRAMES = ["BASE_C", "POS_ACT"]
FRAME_IDS = ["world", "base_link"]
CHILD_FRAME_IDS = ["base_link", "tcp_link"]


def list_to_tf(frame_id: str, child_frame_id: str, pos: List[float]) -> TransformStamped:
    tfs = TransformStamped()

    tfs.header.stamp = rospy.Time.now()
    tfs.header.frame_id = frame_id
    tfs.child_frame_id = child_frame_id
    tfs.transform.translation.x = pos[0] / 1000
    tfs.transform.translation.y = pos[1] / 1000
    tfs.transform.translation.z = pos[2] / 1000
    tfs.transform.rotation.x = pos[3]
    tfs.transform.rotation.y = pos[4]
    tfs.transform.rotation.z = pos[5]
    tfs.transform.rotation.w = pos[6]

    return tfs


class OsvNode:
    def __init__(self, name: str, ip: str, port: int) -> None:

        self.name = name
        self.client = OpenShowVar(ip=ip, port=port)
        self.tf_br = tf2_ros.TransformBroadcaster()
        self.do_pub = rospy.Publisher("/DO_state", Bool, queue_size=10)

    def run(self):
        self.do_pub.publish(self.read_do(DO))

        for i in range(len(FRAMES)):
            frame = self.read_frame(FRAMES[i], FRAME_IDS[i], CHILD_FRAME_IDS[i])
            if frame is not None:
                self.tf_br.sendTransform(frame)

    def read_do(self, do: int, debug: bool = False) -> bool:

        var = (self.client.read(f"$OUT[{do}]", debug=debug)).decode()

        if var == "TRUE":
            return True
        if var == "FALSE":
            return False

        return rospy.logerr(f"{self.name}: error reading DO state")

    def read_frame(self, name: str, frame_id: str, child_frame_id: str, debug: bool = False
                   ) -> Optional[TransformStamped]:
        var = (self.client.read(f"${name}", debug=debug)).decode()

        if not len(var) > 10:
            rospy.loginfo(f"{self.name}: unable to read frame, no main run")
            return None

        var = var.strip("{}").replace(",", "").split(" ")
        pos = list(map(float, [var[2], var[4], var[6], var[8], var[10], var[12]]))

        quat = tf.transformations.quaternion_from_euler(
            math.radians(pos[3]), math.radians(pos[4]), math.radians(pos[5]), "rzyx"
        )

        frame = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]

        # print(f"{child_frame_id}{frame}")

        tfs = list_to_tf(frame_id, child_frame_id, frame)

        return tfs


def main():
    rospy.init_node("osv")
    osv = OsvNode(rospy.get_name(), "192.168.1.100", 7000)
    if not osv.client.can_connect:
        rospy.logerr(f"{osv.name}: cannot connect to KVP server")
        return
    rospy.loginfo(f"{osv.name}: connected to KVP server")
    # r = rospy.Rate(1)
    while not rospy.is_shutdown():
        osv.run()
        # r.sleep()


if __name__ == "__main__":
    main()
