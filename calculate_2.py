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

directory = './results/T2'


def calc_values(paths):
	
	#print(paths)
	dfs = []
	for each in paths:
		print(each)
		dfs.append(pd.read_csv(each, header = None))
	final_df = pd.concat(dfs)
	
	header = []
	for index, content in enumerate(range(len(final_df.columns)-6)):
		header.append('target_' + str(index))
		
	header = header + ['algo', 'handover', 'targets', 'prediction', 'variable', 'time']
	final_df.columns = header
	
	#print(final_df)

	return final_df

	
def save_values(df, handover, prediction, variable, targets, name):
	filtered = df[(df['handover'] == handover) & (df['prediction'] == prediction) & (df['variable'] == variable) & (df['targets'] == targets)]
		
	algo_list = []
	for each in filtered['algo'].unique():
		algo_filtered = filtered[filtered['algo'] == each]
		row_averages = []
		for index, row in algo_filtered.iterrows():
			flat_list = row.values.flatten()
			flat_list = [x for x in flat_list if str(x) != 'nan']
			row_average = mean(flat_list[:-6])
			row_averages.append(row_average)
		algo_mean = mean(row_averages)
		try:
			algo_std = stdev(row_averages)
		except: 
			algo_std = 0
			
		try:	
			algo_error = 1.96 * algo_std / math.sqrt(len(row_averages))/algo_mean
		except:
			algo_error = 0
			
			
		results_list = [name, targets, each, len(row_averages), "{:.2f}".format(algo_mean), "{:.2f}".format(algo_std), "{:.2f}".format(algo_error)]
		algo_list.append(results_list)
	return(algo_list)

def flatten(l):
    return [item for sublist in l for item in sublist]


dirs = [x[0] for x in os.walk(directory)]
dirs = [k for k in dirs if 'BAD' not in k]
dirs = [k for k in dirs if 'DONE' not in k]
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
	
	#print(visit_paths, time_paths)
	#print('visit')
	visit_values = calc_values(visit_paths)
	#print('time')
	time_values = calc_values(time_paths)
	
	#print(test_name, each, len(row_averages), "{:.2f}".format(algo_mean), "{:.2f}".format(algo_std), "{:.2f}".format(algo_error))
	header = ['Test Name', 'targets', 'Algo', 'Sample Size', 'Average', 'Stdev', '% Err']
	value_list = []
	
	value_list.append(save_values(visit_values, True, True, False, 20, '20_visits'))
	value_list.append(save_values(time_values, True, True, False, 20, '20_time'))
	
	value_list.append(save_values(visit_values, True, True, False, 30, '30_visits'))
	value_list.append(save_values(time_values, True, True, False, 30, '30_time'))
	
	value_list.append(save_values(visit_values, True, True, False, 40, '40_visits'))
	value_list.append(save_values(time_values, True, True, False, 40, '40_time'))
	
	value_list.append(save_values(visit_values, True, True, False, 60, '60_visits'))
	value_list.append(save_values(time_values, True, True, False, 60, '60_time'))
	
	value_list.append(save_values(visit_values, True, True, False, 80, '80_visits'))
	value_list.append(save_values(time_values, True, True, False, 80, '80_time'))
	
	value_list.append(save_values(visit_values, True, True, False, 100, '100_visits'))
	value_list.append(save_values(time_values, True, True, False, 100, '100_time'))
	
	#print(value_list)
	df = pd.DataFrame(flatten(value_list), columns = header)
	print(df)
	df.to_csv('./final_scaling.csv')
	
	
	
	
	

	
path = './results/time1.csv'
#returns average, std, error from folder

	
