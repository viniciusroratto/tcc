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
from tf.transformations import euler_from_quaternion
#from math import atan2
import os
import glob






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


time_table = pd.DataFrame(columns = targets)
time_table.to_csv('./distances.csv', index = False)


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
            
@timeit        
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
		tick = rospy.get_time()
		for each in targets:
			x = uniform(-target_max_speed, target_max_speed)
			y = uniform(-target_max_speed, target_max_speed)
			speeds.append([x,y])
        
		model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
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
		tock = rospy.get_time()
		
		while ((tock - tick) < 1.0):
			rospy.sleep(0.01)
			tock = rospy.get_time()
		print('tempo_movimento:', tock - tick)
					

	
def get_distances(points):
	points_coordinate = np.array(list(points))
	distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
	return distance_matrix, points_coordinate
	
@timeit
def genalg(num_points, distance_matrix, points_coordinate, size_pop, max_iter, mode, n,x):

	def cal_total_distance(routine) :
		'''The objective function. input routine, return total distance. cal_total_distance(np.arange(num_points)) '''
		num_points, = routine.shape
		soma = sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
		return soma


	# %% do GA
	set_run_mode(cal_total_distance, mode) #('common', 'multithreading', 'multiprocessing')
	ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=size_pop, max_iter=max_iter, prob_mut=1)
	best_points, best_distance = ga_tsp.run()

	'''
	fig, ax = plt.subplots(1, 2)
	plt.title(x + " " + str(n).zfill(2)  + ' = ' + str(int(min(ga_tsp.generation_best_Y))), loc='left')
	best_points_ = np.concatenate([best_points, [best_points[0]]])
	best_points_coordinate = points_coordinate[best_points_, :]
	ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
	ax[1].plot(ga_tsp.generation_best_Y)
	plt.savefig("./results/GA_" + str(x)+ "_" + str(n).zfill(2)  + ".jpg")
	plt.close('all')
	'''
	print(x, n, min(ga_tsp.generation_best_Y))


	
@timeit
def antcolony(num_points, distance_matrix, points_coordinate, size_pop, max_iter, mode, n, x):


	uav_points = get_points([x])
	points_coordinate = np.concatenate([points_coordinate, np.array(list(uav_points))])
	

	def cal_total_distance(routine) :
		'''The objective function. input routine, return total distance. cal_total_distance(np.arange(num_points)) '''
		num_points, = routine.shape
		soma = sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
		return soma

	# %% Do ACA
	set_run_mode(cal_total_distance, mode) #('common', 'multithreading', 'multiprocessing')
	aca = ACA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=size_pop, max_iter=max_iter, distance_matrix=distance_matrix)
	best_x, best_y = aca.run()

	'''
	fig, ax = plt.subplots(1, 2)
	plt.title(x + " " + str(n).zfill(2)  + ' = ' + str(int(min(aca.y_best_history))), loc='left')
	best_points_ = np.concatenate([best_x, [best_x[0]]])
	best_points_coordinate = points_coordinate[best_points_, :]
	ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
	pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
	plt.savefig("./results/ACA_" + str(x) + "_" + str(n).zfill(2)  + ".jpg")
	plt.close('all')
	'''
	print(x, n, min(aca.y_best_history))


def clean_results ():
	files = glob.glob('./results/*')
	if len(files) > 0:
		for f in files:
			os.remove(f)

def first_auction ():
	
	target_list = get_points(targets)
	uav_list = get_points(uavs)
	
	auctioned = []
	for each in uavs:
		auctioned.append([])
		
	for xi, x in enumerate(targets):
		values = []
		for yi, y in enumerate(uav_list):
			values.append(1 / ((math.sqrt(((y[0]-target_list[xi][0])**2) + (y[1]-target_list[xi][1])**2 ) * 0.1*(len(auctioned[yi]) + 1))))
		index = values.index(max(values))
		auctioned[index].append(target_list[xi])
	
	nums = []	
	for each in auctioned:
		nums.append(len(each))
			
	return auctioned
			
def smaller_distance(target, uavs):

	uav_points = get_points(uavs)
	target_list = []
	target_list.append(target)
	target_points = get_points(target_list)[0]
	
	for each in uav_points:
		distances = []
		distances.append(((math.sqrt(((target_points[0]-each[0])**2) + (target_points[1]-each[1])**2 ))))

	smaller_distance = min(distances)
	return smaller_distance
		

def monitor_distances(targets, uavs):

	while True:
		tick = rospy.get_time()
		uav_points = get_points(uavs)
		targets_points = get_points(targets)
		smaller_distances = []
		
		for x in targets_points:
			distances = []
			for y in uav_points:
				distances.append(((math.sqrt(((x[0]-y[0])**2) + (x[1]-y[1])**2 ))))
			smaller_distances.append(min(distances))
			
		distance_dict = dict(zip(targets, smaller_distances))
		df = time_table.append(distance_dict, ignore_index = True)
		df.to_csv('./distances.csv', mode='a', header=False, index= False )
		tock = rospy.get_time()
		
		while ((tock - tick) < 1.0):
			rospy.sleep(0.01)
			tock = rospy.get_time()
			
def move_uavs():
	x = 0.0
	y = 0.0 
	theta = 0.0

	def newOdom(msg, x, y, theta):

		x = msg.pose.pose.position.x
		y = msg.pose.pose.position.y

		rot_q = msg.pose.pose.orientation
		(roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

	rospy.init_node("speed_controller")

	sub = rospy.Subscriber("/odometry/filtered", Odometry, newOdom)
	pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

	speed = Twist()

	r = rospy.Rate(4)

	goal = Point()
	goal.x = 5
	goal.y = 5

	while not rospy.is_shutdown():
		inc_x = goal.x -x
		inc_y = goal.y -y

		angle_to_goal = atan2(inc_y, inc_x)

		if abs(angle_to_goal - theta) > 0.1:
		    speed.linear.x = 0.0
		    speed.angular.z = 0.3
		else:
		    speed.linear.x = 0.5
		    speed.angular.z = 0.0

		pub.publish(speed)
		r.sleep()  


def main():
	
	algo = 0
	clean_results()
	uav_dict = dict(zip(uavs, get_points(uavs)))
	#targets_dict = dict(zip(targets, get_points([targets])))
	
	world_size = 100
	target_max_speed = 2
	target_z = 0
	uav_max_speed = 20
	uav_z = 3
	
	reset(0, boxes, uavs, targets, world_size)

	size_pop = 26
	max_iter = 100
	mode = 'common' #('common', 'multithreading', 'multiprocessing', 'vectorization', 'cached')
	

	#_thread.start_new_thread(targets_movement, (targets, target_max_speed,world_size,target_z))
	#_thread.start_new_thread(monitor_distances, (targets, uavs))

	auctioned_targets = first_auction()	
	n = 0	
	while True:
		
		n = n + 1
		for xi, x in enumerate(uavs):
		
			num_points = len(auctioned_targets[xi])
				
			distance_matrix, points_coordinate = get_distances(auctioned_targets[xi])
			
			if(algo == 0):
				genalg(num_points, distance_matrix, points_coordinate, size_pop, max_iter, mode, n, x)
			if(algo == 1):
				antcolony(num_points, distance_matrix, points_coordinate, size_pop, int(max_iter/10), mode, n, x)
		auctioned_targets = first_auction()

	



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("Basic Main Error")
        
       
