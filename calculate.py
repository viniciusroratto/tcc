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
from os import listdir
from os.path import isfile, join
import glob
from datetime import datetime
from sko.SA import SA_TSP
from sko.IA import IA_TSP
import threading
from statistics import mean, stdev

def calc_values(paths):
	
	#print(paths)
	dfs = []
	for each in paths:
		dfs.append(pd.read_csv(each, header = None))
		
	final_df = pd.concat(dfs)
	#print(final_df.head())
	
	big_list = []
	for column in final_df:
		big_list.append(mean(list(final_df[column])))
	
	avg = mean(big_list)
	
	std = stdev(big_list)
	err = 1.96 * std / math.sqrt(len(big_list))
	
	#print(avg, std, err)
	
	return [avg, std, err]
	

directory = './results'
dirs = [x[0] for x in os.walk(directory)]
dirs = [k for k in dirs if 'BAD' not in k]
#dirs.remove(directory)
dirs.sort()
#print(dirs)
Numbers = []
time_dicts = []
visit_dicts = []

columns = ['title', 'avg_visits', 'avg_time', 'std_visit', 'std_time', 'err_visits', 'err_time']
df = pd.DataFrame(columns = columns)

for each in dirs:
	#print(each)
	onlyfiles = [f for f in listdir(each) if isfile(join(each, f))]
	time = [k for k in onlyfiles if 'time' in k]
	visits = [k for k in onlyfiles if 'visits' in k]

	
	visit_paths = []
	for files in visits:
		path = each + '/' + files
		visit_paths.append(path)
	
	time_paths = []
	for files in time:
		path = each + '/' + files
		time_paths.append(path)
	
	#print('visit')
	visit_values = calc_values(visit_paths)
	#print('time')
	time_values = calc_values(time_paths)
	
	df = df.append({'title': each, 'avg_visits' : visit_values[0], 'avg_time' : time_values[0], 
	'std_visit' : visit_values[1], 'std_time':time_values[1], 'err_visits':visit_values[2], 'err_time':time_values[2]}, ignore_index = True)
print(df)	
	
df.to_csv('./final_results')
	
	

	
path = './results/time1.csv'
#returns average, std, error from folder

	
