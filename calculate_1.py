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
import shutil

directory = './results/T1_DONE'


def calc_values(paths):
	
	#print(paths)
	

	
	dfs = []
	for each in paths:
		if (len(pd.read_csv(each, header = None).columns) > 30):
			filename = each.split('/')[-1]
			new_path = './results/T2/' + filename
			print(each, new_path)
			shutil.move(each, new_path)
		else:
			dfs.append(pd.read_csv(each, header = None))
	final_df = pd.concat(dfs)
	
	header = []
	for index, content in enumerate(range(len(final_df.columns)-5)):
		header.append('target_' + str(index))
		
	header = header + ['time', 'algo', 'handover','prediction', 'variable']
	final_df.columns = header
	
	#print(final_df)

	return final_df

	
def save_values(df, handover, prediction, variable,name):
	filtered = df[(df['handover'] == handover) & (df['prediction'] == prediction) & (df['variable'] == variable)]
		
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
			
			
		results_list = [name, each, len(row_averages), "{:.2f}".format(algo_mean), "{:.2f}".format(algo_std), "{:.2f}".format(algo_error)]
		algo_list.append(results_list)
	return(algo_list)

def flatten(l):
    return [item for sublist in l for item in sublist]


dirs = [x[0] for x in os.walk(directory)]
dirs = [k for k in dirs if 'BAD' not in k]
#dirs = [k for k in dirs if 'DONE' not in k]
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
	header = ['Test Name', 'Algo', 'Sample Size', 'Average', 'Stdev', '% Err']
	value_list = []
	
	value_list.append(save_values(visit_values, False, False, False, 'S1-visit'))
	value_list.append(save_values(time_values, False, False, False, 'S1-time'))
	
	value_list.append(save_values(visit_values, False, True, False, 'S2-visit'))
	value_list.append(save_values(time_values, False, True, False,'S2-time'))
	
	value_list.append(save_values(visit_values, True, False, False,'S3-visit'))
	value_list.append(save_values(time_values, True, False, False, 'S3-time'))
	
	value_list.append(save_values(visit_values, True, True, False,'S4-visit'))
	value_list.append(save_values(time_values, True, True, False, 'S4-time'))
	
	value_list.append(save_values(visit_values, True, False, True,'S5-visit'))
	value_list.append(save_values(time_values, True, False, True, 'S5-time'))
	
	value_list.append(save_values(visit_values, True, True, True, 'S6-visit'))
	value_list.append(save_values(time_values, True, True, True, 'S6-time'))
	
	#print(value_list)
	df = pd.DataFrame(flatten(value_list), columns = header)
	print(df)
	df.to_csv('./final_Bahavior.csv')
	
	
	
	
	

	
path = './results/time1.csv'
#returns average, std, error from folder

	
