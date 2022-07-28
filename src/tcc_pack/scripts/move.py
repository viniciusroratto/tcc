import rospy 
import rospkg 
from gazebo_msgs.msg import ModelState 
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
import os
import glob


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        #print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

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
    x = model_coordinates(name, "").pose.position.x
    y = model_coordinates(name, "").pose.position.y
    z = model_coordinates(name, "").pose.position.z
    move(name, x + speed_x , y + speed_y, z + speed_z)
    	
def get_points(targets):
	
	points = []
	model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
	for each in targets:
		points.append([model_coordinates(each, "").pose.position.x, model_coordinates(each, "").pose.position.y])
	return points
	

def callback(data):
    rospy.loginfo("%s",data.data)

def reset(blocks, boxes, uavs, targets, world_size):
    target_z = 0
    uav_z = 3
    
    for each in boxes:
        if blocks == 1:
            x = randint(-world_size, world_size)
            y = randint(-world_size, world_size)
            move(each, x, y, 0)
        else:
            move(each, world_size+100, world_size+100, 0)
        
        
    for each in uavs:
        x = randint(-world_size, world_size)
        y = randint(-world_size, world_size)
        move(each, x, y, uav_z)
      
    for each in targets:
        x = randint(-world_size, world_size)
        y = randint(-world_size, world_size)
        move(each, x, y, target_z)

def targets_movement(targets, target_max_speed, world_size, target_z):

	model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
	
	speeds = []
	
	while True:
		time.sleep(1)
		for each in targets:
			x = uniform(-target_max_speed, target_max_speed)
			y = uniform(-target_max_speed, target_max_speed)
			speeds.append([x,y])
        
		for idx, each in enumerate(targets):
        
			if(abs(model_coordinates(each, "").pose.position.x) <= world_size and abs(model_coordinates(each, "").pose.position.y) <= world_size):
			
				acelerate_to(each, speeds[idx][0], speeds[idx][1], target_z)
				
			else:
				if (abs(model_coordinates(each, "").pose.position.x) >= world_size):
					speeds[idx][0] = -speeds[idx][0]
					acelerate_to(each, speeds[idx][0], speeds[idx][1], target_z)
					
				else:
					speeds[idx][1] = -speeds[idx][1]
					acelerate_to(each, speeds[idx][0], speeds[idx][1], target_z)
					

	
def get_distances(points):
	points_coordinate = np.array(list(points.values()))
	distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
	return distance_matrix, points_coordinate
	
	
@timeit
def genalg(num_points, distance_matrix, points_coordinate, size_pop, max_iter, mode, n):

	def cal_total_distance(routine) :
		'''The objective function. input routine, return total distance. cal_total_distance(np.arange(num_points)) '''
		num_points, = routine.shape
		soma = sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
		print(soma)
		return soma


	# %% do GA
	set_run_mode(cal_total_distance, mode) #('common', 'multithreading', 'multiprocessing')
	ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=size_pop, max_iter=max_iter, prob_mut=1)
	best_points, best_distance = ga_tsp.run()

	fig, ax = plt.subplots(1, 2)
	best_points_ = np.concatenate([best_points, [best_points[0]]])
	best_points_coordinate = points_coordinate[best_points_, :]
	ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
	ax[1].plot(ga_tsp.generation_best_Y)
	plt.savefig("./results/GA_" + str(n) + ".jpg")


	
@timeit
def antcolony(num_points, distance_matrix, points_coordinate, size_pop, max_iter, mode, n):

	def cal_total_distance(routine) :
		'''The objective function. input routine, return total distance. cal_total_distance(np.arange(num_points)) '''
		num_points, = routine.shape
		soma = sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
		return soma

	# %% Do ACA
	set_run_mode(cal_total_distance, mode) #('common', 'multithreading', 'multiprocessing')
	aca = ACA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=size_pop, max_iter=max_iter, distance_matrix=distance_matrix)
	best_x, best_y = aca.run()

	fig, ax = plt.subplots(1, 2)
	best_points_ = np.concatenate([best_x, [best_x[0]]])
	best_points_coordinate = points_coordinate[best_points_, :]
	ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
	pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
	plt.savefig("./results/ACA_" + str(n) + ".jpg")


def clean_results ():
	files = glob.glob('./results/*')
	if len(files) > 0:
		for f in files:
			os.remove(f)
"""
def auction (uavs, targets):
	
	target_list = targets
	
	for x,y in enumerate(uavs):
		distances_list = []
		for q in target_list:
			distances_list.append(math.dist(uavs[x],q))
		targets_dict = dict(zip(target_list, distances_list))
			
"""


models = ["UAV_00", "UAV_01", "UAV_02", "UAV_03", "UAV_04",
          "target_00", "target_01", "target_02", "target_03", "target_04",
          "target_05", "target_06", "target_07", "target_08", "target_09", "target_10",
          "target_11", "target_12", "target_13", "target_14", "target_15", "target_16",
          "target_17", "target_18", "target_19", "target_20", "target_21", "target_22",
          "target_23", "target_24",
          "box_00", "box_01", "box_02", "box_03", "box_04",
          "box_05", "box_06", "box_07", "box_08", "box_09",
          "box_10", "box_11", "box_12", "box_13", "box_14",
          "box_15", "box_16", "box_17", "box_18", "box_19"]
 
uavs = list(filter(lambda k: 'UAV' in k, models))
targets = list(filter(lambda k: 'target' in k, models))
boxes = list(filter(lambda k: 'box' in k, models))   


      


def main():
	
	clean_results()
	uav_dict = dict(zip(uavs, get_points(uavs)))
	targets_dict = dict(zip(targets, get_points(targets)))
	
	world_size = 100
	target_max_speed = 1
	target_z = 0
	uav_max_speed = 20
	uav_z = 3
	
	reset(0, boxes, uavs, targets, world_size)

	size_pop = 26
	max_iter = 100
	mode = 'common' #('common', 'multithreading', 'multiprocessing', 'vectorization', 'cached')
	

	_thread.start_new_thread(targets_movement, (targets, target_max_speed,world_size,target_z ))

		
	n = 0	
	while n < 20:
		n = n + 1
		targets_dict = dict(zip(targets, get_points(targets)))
		uav_dict.update(targets_dict)
		num_points = len(uav_dict)
		distance_matrix, points_coordinate = get_distances(uav_dict)
		genalg(num_points, distance_matrix, points_coordinate, size_pop, max_iter, mode, n)
		antcolony(num_points, distance_matrix, points_coordinate, size_pop, max_iter, mode, n)
		print(n)
		time.sleep(5)

	



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("Basic Main Error")
        
       
