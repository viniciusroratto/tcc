import rospy 
import rospkg 
from gazebo_msgs.msg import ModelState 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from std_srvs.srv import Empty
from std_msgs.msg import String
from random import seed
from random import randint, uniform
import _thread
import time
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import pandas as pd
from functools import wraps
from sko.ACA import ACA_TSP
from sko.GA import GA_TSP
from sko.tools import set_run_mode
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import atan2
import os
import glob
from datetime import datetime
from sko.SA import SA_TSP
from sko.IA import IA_TSP
import threading


models = ["UAV_00", "UAV_01", "UAV_02", "UAV_03", "UAV_04",
          "target_00", "target_01", "target_02", "target_03", "target_04",
          "target_05", "target_06", "target_07", "target_08", "target_09", "target_10",
          "target_11", "target_12", "target_13", "target_14", "target_15", "target_16",
          "target_17", "target_18", "target_19",
          "box_00", "box_01", "box_02", "box_03", "box_04",
          "box_05", "box_06", "box_07", "box_08", "box_09",
          "box_10", "box_11", "box_12", "box_13", "box_14",
          "box_15", "box_16", "box_17", "box_18", "box_19"]
 
uavs = list(filter(lambda k: 'UAV' in k, models))
targets = list(filter(lambda k: 'target' in k, models))
boxes = list(filter(lambda k: 'box' in k, models))   


world_size = 250
speeds = []

time_table = pd.DataFrame(columns = targets)

dt = datetime.now()
ts = datetime.timestamp(dt)
#time_table.to_csv('./results/distances_' + str(ts) +'.csv', index = False)

auctioned_targets = []
for each in uavs:
	auctioned_targets.append([])
	
results  = []
for each in targets:
	results.append([])
	
visits_table = pd.DataFrame(columns = targets)
visits_table.to_csv('./results/visits.csv', index = False)

time_table = pd.DataFrame(columns = targets)
time_table.to_csv('./results/time.csv', index = False)
