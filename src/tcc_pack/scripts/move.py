import rospy 
import rospkg 
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import time





def move(name, x,y,z):
    rospy.init_node('set_pose')

    state_msg = ModelState()
    state_msg.model_name = name
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    state_msg.pose.position.z = z
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = 0
    state_msg.pose.orientation.w = 0

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( state_msg )

    except rospy.ServiceException as e:
        print ("Service call failed: %s" % e)


def main():

    move('UAV_00', -5, -5, 5)
    time.sleep(2)
    move('UAV_00', -7, -7, 7)
    time.sleep(2)
    move('UAV_00', -10, -10, 10)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
