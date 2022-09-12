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


for x in range(100):
	for each in [0,1,2,3]:
		try:
			os.system('clear')
			print(x, each, datetime.now())
			os.system('rosrun tcc_pack move.py ' + str(each))
		except:
			print('terminou')

 

