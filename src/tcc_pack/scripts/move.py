import rospy 
import rospkg 
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from std_msgs.msg import String
from random import seed
from random import randint

import time

models = ["UAV_00", 
          "target_00", "target_01", "target_02",
          "box_00", "box_01", "box_02", "box_03", "box_04",
          "box_05", "box_06", "box_07", "box_08", "box_09",
          "box_10", "box_11", "box_12", "box_13", "box_14",
          "box_15", "box_16", "box_17", "box_18", "box_19",]
          
uavs = list(filter(lambda k: 'UAV' in k, models))
targets = list(filter(lambda k: 'target' in k, models))
boxes = list(filter(lambda k: 'box' in k, models))
world_size = 50


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
        
        
        
def acelerate_to(name, speed_x, speed_y, speed_z):
    model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
    x = model_coordinates("UAV_00", "").pose.position.x
    y = model_coordinates("UAV_00", "").pose.position.y
    z = model_coordinates("UAV_00", "").pose.position.z
    move(name, x + speed_x , y + speed, x + speed_z)
    	


def callback(data):
    rospy.loginfo("%s",data.data)

def init(blocks = False):
    
    if(blocks == False):
        for each in boxes
        


def main():
    value = randint(-world_size, world_size)
    print(value)

    #while True:
     #   time.sleep(1)
     #  acelerate_to('UAV_00', -1, -1, 0)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("Basic Main Error")
