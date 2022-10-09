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
import sys




models = ["UAV_00", "UAV_01", "UAV_02", "UAV_03", "UAV_04",
          "target_00", "target_01", "target_02", "target_03", "target_04", "target_05", "target_06", "target_07", "target_08", "target_09",
          "target_10", "target_11", "target_12", "target_13", "target_14", "target_15", "target_16", "target_17", "target_18", "target_19", 
          "target_20", "target_21", "target_22", "target_23", "target_24", "target_25", "target_26", "target_27", "target_28", "target_29",
          "target_30", "target_31", "target_32", "target_33", "target_34", "target_35", "target_36", "target_37", "target_38", "target_39", 
          "target_40", "target_41", "target_42", "target_43", "target_44", "target_45", "target_46", "target_47", "target_48", "target_49", 
          "target_50", "target_51", "target_52", "target_53", "target_54", "target_55", "target_56", "target_57", "target_58", "target_59", 
          "target_60", "target_61", "target_62", "target_63", "target_64", "target_65", "target_66", "target_67", "target_68", "target_69", 
          "target_70", "target_71", "target_72", "target_73", "target_74", "target_75", "target_76", "target_77", "target_78", "target_79", 
          "target_80", "target_81", "target_82", "target_83", "target_84", "target_85", "target_86", "target_87", "target_88", "target_89", 
          "target_90", "target_91", "target_92", "target_93", "target_94", "target_95", "target_96", "target_97", "target_98", "target_99",
          "box_00", "box_01", "box_02", "box_03", "box_04",
          "box_05", "box_06", "box_07", "box_08", "box_09",
          "box_10", "box_11", "box_12", "box_13", "box_14",
          "box_15", "box_16", "box_17", "box_18", "box_19"]
 
uavs = list(filter(lambda k: 'UAV' in k, models))
targets = list(filter(lambda k: 'target' in k, models))
boxes = list(filter(lambda k: 'box' in k, models))   


world_size = [750, 25] 
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
#visits_table.to_csv('./results/visits.csv', index = False)

time_table = pd.DataFrame(columns = targets)
#time_table.to_csv('./results/time.csv', index = False)

waiting_list = [] #(target,xi)

sem = threading.Semaphore(1)

internal_time = 0

def general_auction():
	global sem
	global waiting_list
	global auctioned_targets
	
	r = rospy.Rate(1)
	
	while True:

	
		if len(waiting_list) > 0:
			sem.acquire()
			
			target_list = []
			for each in waiting_list:
				target_list.append(each[0])
				
			target_list = get_points(target_list)
			uav_list = get_points(uavs)
				
			for xi, x in enumerate(waiting_list):
				values = []
				target = x[0]
				zi = x[1]
				
				position = targets.index(target)
				
				target_speed = np.linalg.norm(np.array(speeds[position]))
						
				for yi, y in enumerate(uav_list):	
					#print(yi, zi)					
					if yi != zi:
						try:
							values.append(1 / (target_speed * (math.sqrt(((y[0]-target_list[xi][0])**2) + 1 + (y[1]-target_list[xi][1])**2 ) + 25*(len(auctioned_targets[yi])))))
						except:
							values.append(0)
					else:
						values.append(0)
				#print(values)
				index = values.index(max(values))
				auctioned_targets[index].append(target)
				
				
				nums = []	
				for each in auctioned_targets:
					nums.append(len(each))
				print(nums)
			
				waiting_list = []			
				sem.release()
		r.sleep()


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

def turn(name ,x,y,z, roll, pitch, yaw):

	try:
		rospy.init_node('set_pose')

		state_msg = ModelState()
	except:
		print('Set Pose - Turn')
    
	state_msg.model_name = name
	state_msg.pose.position.x = x
	state_msg.pose.position.y = y
	state_msg.pose.position.z = z
    
	[xo,yo, zo, wo ] = quaternion_from_euler(roll, pitch, yaw)
    
	state_msg.pose.orientation.x = xo
	state_msg.pose.orientation.y = yo
	state_msg.pose.orientation.z = zo
	state_msg.pose.orientation.w = wo

	try:
		rospy.wait_for_service('/gazebo/set_model_state')
		set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
		resp = set_state( state_msg )
	except rospy.ServiceException as e:
		print ("Service call failed: %s" % e)

'''        
def turn_to(name, speed_z):
    model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
    z = model_coordinates(name, "").pose.orientation.z
    turn(name, z + speed_z)
'''
#@timeit
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
        resp = set_state(state_msg)

    except rospy.ServiceException as e:
        print ("Service call failed: %s" % e)    
            
#@timeit        
def acelerate_to(name, speed_x, speed_y, speed_z):

	try:
		model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
		x = model_coordinates(name, "").pose.position.x
		y = model_coordinates(name, "").pose.position.y
		z = model_coordinates(name, "").pose.position.z
		move(name, x + speed_x , y + speed_y, z + speed_z)
	except:
		print('aceleration_fail')
       	
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
            x = randint(-world_size[0], world_size[0])
            y = randint(-world_size[1], world_size[1])
            move(each, x, y, 0)
        else:
            move(each, world_size[0] +1000, world_size[1] +1000, 0)
        
        
    for each in uavs:
        x = randint(-world_size[0], world_size[0])
        y = randint(-world_size[1], world_size[1])
        move(each, x, y, uav_z)
      
    for each in targets:
        x = randint(-world_size[0], world_size[0])
        y = randint(-world_size[1], world_size[1])
        move(each, x, y, target_z)

def targets_movement(targets, target_max_speed, world_size, target_z, push):
	
	global internal_time
	
	try:
		model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
	except:
		print('Failure at getting coordinates')
		
	global speeds
	r = rospy.Rate(1)
	tick = rospy.get_time()
	
	while not rospy.is_shutdown():
		internal_time = internal_time + 1
		for each in targets:
			x = uniform(-target_max_speed, target_max_speed)
			y = 0
			speeds.append([x,y])
		
		tock = rospy.get_time()	
		if (push == True and (tock - tick) > 20):
			tick = rospy.get_time()
			
			for index, each in enumerate(targets):
				xi = uniform(-0.5, 0.5)
				yi = 0
				speeds[index][0] = speeds[index][0] + xi
				speeds[index][1] = speeds[index][1] + yi
		try:
			model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
			
			for idx, each in enumerate(targets):
				if(abs(model_coordinates(each, "").pose.position.x) <= world_size[0] and abs(model_coordinates(each, "").pose.position.y) <= world_size[1]):
					acelerate_to(each, speeds[idx][0], speeds[idx][1], target_z)
					
				else:
					if (abs(model_coordinates(each, "").pose.position.x) >= world_size[0]):
						speeds[idx][0] = -speeds[idx][0]
						acelerate_to(each, speeds[idx][0], speeds[idx][1], target_z)
						
					else:
						speeds[idx][1] = -speeds[idx][1]
						acelerate_to(each, speeds[idx][0], speeds[idx][1], target_z)
		
		except:
			print('failure getting coordinates')
		r.sleep()			

	
def get_distances(points):
	points_coordinate = np.array(points)
	#print(points_coordinate, type(points_coordinate))
	distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
	return distance_matrix, points_coordinate

@timeit
def ia(num_points, targets, points_coordinate, size_pop, max_iter, mode, x):

	uav_points = get_points([x])[0]
	new_targets = targets
	new_targets.append(uav_points)
	
	distance_matrix, points_coordinate = get_distances(new_targets)
	num_points = len(points_coordinate)
	
	
	def cal_total_distance(routine) :
		'''The objective function. input routine, return total distance. cal_total_distance(np.arange(num_points)) '''
		num_points, = routine.shape
		soma = sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
		return soma

	set_run_mode(cal_total_distance, mode) #('common', 'multithreading', 'multiprocessing')
	ia_tsp = IA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=size_pop, max_iter=max_iter, prob_mut=0.2, T=0.7, alpha=0.95)
	best_points, best_distance = ia_tsp.run()
	best_points_coordinate = list(points_coordinate[best_points, :])
	
	while uav_points != list(best_points_coordinate[0]):
		best_points_coordinate.append(best_points_coordinate.pop(0))

	return best_points_coordinate, best_distance



@timeit
def sa(num_points, targets, points_coordinate, size_pop, max_iter, mode, x):

	uav_points = get_points([x])[0]
	new_targets = targets
	new_targets.append(uav_points)
	
	distance_matrix, points_coordinate = get_distances(new_targets)
	num_points = len(points_coordinate)
	
	
	def cal_total_distance(routine) :
		'''The objective function. input routine, return total distance. cal_total_distance(np.arange(num_points)) '''
		num_points, = routine.shape
		soma = sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
		return soma

	set_run_mode(cal_total_distance, mode) #('common', 'multithreading', 'multiprocessing')
	sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=250, T_min=50, L=5)
	best_points, best_distance = sa_tsp.run()
	best_points_coordinate = list(points_coordinate[best_points, :])
	
	while uav_points != list(best_points_coordinate[0]):
		best_points_coordinate.append(best_points_coordinate.pop(0))

	return best_points_coordinate, best_distance

	
@timeit
def genalg(num_points, targets, points_coordinate, size_pop, max_iter, mode, x):

	uav_points = get_points([x])[0]
	new_targets = targets
	new_targets.append(uav_points)
	
	#print(new_targets) 
	distance_matrix, points_coordinate = get_distances(new_targets)
	num_points = len(points_coordinate)
	
	
	def cal_total_distance(routine) :
		'''The objective function. input routine, return total distance. cal_total_distance(np.arange(num_points)) '''
		num_points, = routine.shape
		soma = sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
		return soma


	# %% do GA
	set_run_mode(cal_total_distance, mode) #('common', 'multithreading', 'multiprocessing')
	ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=size_pop, max_iter=max_iter, prob_mut=1)
	best_points, best_distance = ga_tsp.run()
	best_points_coordinate = list(points_coordinate[best_points, :])
	
	while uav_points != list(best_points_coordinate[0]):
		best_points_coordinate.append(best_points_coordinate.pop(0))

	return best_points_coordinate, best_distance


	
@timeit
def antcolony(num_points, targets, points_coordinate, size_pop, max_iter, mode, x):

	uav_points = get_points([x])[0]
	new_targets = targets
	new_targets.append(uav_points)
	
	
	distance_matrix, points_coordinate = get_distances(new_targets)
	num_points = len(points_coordinate)
	

	def cal_total_distance(routine):
		num_points, = routine.shape
		return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

	# %% Do ACA
	set_run_mode(cal_total_distance, mode) #('common', 'multithreading', 'multiprocessing')
	aca = ACA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=size_pop, max_iter=int(max_iter), distance_matrix=distance_matrix)
	best_x, best_y = aca.run()
	
	best_points_coordinate = list(points_coordinate[best_x, :])
	
	while uav_points != list(best_points_coordinate[0]):
		best_points_coordinate.append(best_points_coordinate.pop(0))
	
	#print('best_points', len(best_points_coordinate))
	return best_points_coordinate, best_y


def clean_results ():
	files = glob.glob('./results/*')
	if len(files) > 0:
		for f in files:
			os.remove(f)

def first_auction (auctioned):
	
	target_list = get_points(targets)
	uav_list = get_points(uavs)
		
	for xi, x in enumerate(targets):
		values = []
		target_speed = np.linalg.norm(np.array(speeds[xi]))
		
		for yi, y in enumerate(uav_list):
			values.append(1 / (target_speed * (math.sqrt(((y[0]-target_list[xi][0])**2) + 
					(y[1]-target_list[xi][1])**2 ) + 1 + 25 *(len(auctioned[yi])))))
		index = values.index(max(values))
		auctioned[index].append(targets[xi])
	
	nums = []	
	for each in auctioned:
		nums.append(len(each))
	print(nums)
	#print(auctioned)		
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
		

def monitor_distances(targets, uavs, ts):

	r = rospy.Rate(1)

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
		df.to_csv('./results/distances_'+ str(ts) + '.csv', mode='a', header=False, index= False )
		tock = rospy.get_time()
		r.sleep()


def uav_move(goal_x, goal_y, uav):

	x = 0.0
	y = 0.0
	z = 20.0
	roll = 0.0
	pitch = 0.0
	yaw = 0.0
	sp = 20.0
	
	def newOdom(msg):
		nonlocal x
		nonlocal y
		nonlocal z
		
		nonlocal yaw

		x = msg.pose.pose.position.x
		y = msg.pose.pose.position.y
		z = msg.pose.pose.position.z

		rot_q = msg.pose.pose.orientation
		(roll, pitch, yaw) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w]) #rad
	
	try:
		sub = rospy.Subscriber("/odom_" + uav, Odometry, newOdom)
		pub = rospy.Publisher("/cmd_vel_" + uav, Twist, queue_size = 1)
	except:
		print('Publishing Error')

	speed = Twist()
	speed.linear.x = 0
	speed.linear.y = 0
	
	try:
		pub.publish(speed)
		rospy.wait_for_message("/odom_" + uav, Odometry)
	except:
		print('Publishing Error')	
	
	
	r = rospy.Rate(20)

	goal = Point()
	goal.x = goal_x
	goal.y = goal_y
	#print(type(goal.x), type(x), type(goal.y), type(y))
	distance = math.sqrt(((goal_x - x)**2) + (y - goal_y)**2 )
	
	speed.linear.x = 0.0
	speed.angular.z = 0.0

	while (distance > 5):
		
		#rospy.Subscriber("/odom_" + uav, Odometry, newOdom)
		inc_x = goal.x -x
		inc_y = goal.y -y

		angle_to_goal = atan2(inc_y, inc_x) #radianos
			
		if abs(angle_to_goal - yaw) > 0.2:
			speed.linear.x = 0.0
			speed.linear.y = 0.0
			turn(uav,x,y,z, roll, pitch, angle_to_goal)

		else:
		    speed.linear.x = sp
		    speed.linear.y = 0.0
		    speed.angular.z = 0.0
			

		try:
			pub.publish(speed)
		except:
			print('Publishing Error')
		
		distance = math.sqrt(((goal.x - x)**2) + (y - goal.y)**2 )
		#print(uav, [goal.x, goal.y], distance, int(yaw - angle_to_goal) )
		
		if distance < sp:
			sp = sp/2
		r.sleep()
		
	speed.linear.x = 0.0
	speed.linear.y = 0.0
	speed.angular.z = 0.0
	
	try:
		pub.publish(speed)
	except:
		print('Publishing Error')

def get_distance(x,y,x1,y1):
	arraya = np.array((x,y))
	arrayb = np.array((x1,y1))
	return np.linalg.norm(arraya-arrayb)

def get_offset(target_list, index, tick, x):
	target_name = target_list[index]
	speed_index = targets.index(target_name)
	xv = speeds[speed_index][0]
	yv = speeds[speed_index][1]
	
	uav_points = get_points([x])[0]
	target_points = get_points([target_name])[0]
	
	dist = get_distance(uav_points[0], uav_points[1], target_points[0], target_points[1])
	extra_time = dist/20

	#tock = rospy.get_time()
	tock = internal_time
	offsetx = ((tock-tick) + extra_time) * xv
	offsety = ((tock-tick) + extra_time) * yv
	
	return offsetx, offsety	
	
def send_data(target, index, tick):

	global results
	
	position = targets.index(target)
	tock = rospy.get_time()
	#print(tock-tick)
	results[position].append(tock-tick)
	
def get_random_points():
	x = randint(-world_size[0], world_size[0])
	y = randint(-world_size[1], world_size[1])
	return [x,y]


	
	
def fly(xi, x, algo, tick, delivery, prediction):	
	size_pop = 26
	max_iter = 10
	mode = 'common' #('common', 'multithreading', 'multiprocessing', 'vectorization', 'cached')
	global auctioned_targets
	global sem
	run_algo = True
	auctioned_targets_updated = auctioned_targets[xi]
	t1 = rospy.get_time()
	#r = rospy.Rate(1)
	limit = 600
	global internal_time
			
	while not rospy.is_shutdown():
	
		if(auctioned_targets_updated != auctioned_targets[xi]):
			auctioned_targets_updated = auctioned_targets[xi]
			run_algo = True
			t1 = rospy.get_time()
			
	
		if(run_algo == True and not len(auctioned_targets_updated)==0):
		
			#print(auctioned_targets[xi])
			target_names = auctioned_targets_updated
			targets_points = get_points(target_names)
			
			num_points = len(targets_points)	
			distance_matrix, points_coordinate = get_distances(targets_points)

			if(algo == 0):
				points, distance = genalg(num_points, targets_points, points_coordinate, size_pop, max_iter, mode, x)
			if(algo == 1):
				points, distance = antcolony(num_points, targets_points, points_coordinate, size_pop, int(max_iter), mode, x)
			if(algo == 2):
				points, distance = sa(num_points, targets_points, points_coordinate, size_pop, int(max_iter), mode, x)
			if(algo == 3):
				points, distance = ia(num_points, targets_points, points_coordinate, size_pop, int(max_iter), mode, x)
				
			run_algo = True
			
		
		#time = rospy.get_time()
		time = internal_time
		
		target_list = []
		if (len(auctioned_targets_updated)==0):
			target_list.append(get_random_points())
			target_list.append(get_random_points())
			target_list.append(get_random_points())
		else:
			for each in points[1:]:
				position = targets_points.index(list(each))
				#print(position, each, targets_points)
				target_list.append(target_names[position])
				last_target = target_list[0]
		
			last_target = target_list[-1]
			
			reverse = target_list[:-1]
			reverse.reverse()
			target_list.extend(reverse)
		
		final_points = []
		for each in target_list:
			final_points.append(get_points([each])[0])
			
				
		if (delivery == False):
			for index, alvo in enumerate(final_points):
				#print(int(alvo[0]), int(alvo[1]))
				if(prediction == True):				
					offsetx, offsety = get_offset(target_list, index, time,x)
					
					realx = alvo[0] + offsetx
					realy = alvo[1] + offsety
					uav_move(realx, realy, x)
										
				else:
					realx = alvo[0]
					realy = alvo[1]
					uav_move(realx, realy, x)
					
				final_position = get_points([target_list[index]])[0]	
				distance = get_distance(realx,realy,final_position[0],final_position[1])
				#print(distance, [realx,realy], final_position)
				if(auctioned_targets[xi] != 0 and distance < 20):
					send_data(target_list[index], index, tick)
				
			tock = rospy.get_time()
			#print(x, str(2*distance/(tock-tick)))
		else:
			
			if distance < limit or (distance > limit and rospy.get_time() - t1 < 10) or len(auctioned_targets_updated) <= 2:
				#print(distance, rospy.get_time() - t1)
				for index, alvo in enumerate(final_points):
					#print(int(alvo[0]), int(alvo[1]))
					if(prediction == True):				
						offsetx, offsety = get_offset(target_list, index, time, x)
						
						realx = alvo[0] + offsetx
						realy = alvo[1] + offsety
						
						uav_move(realx, realy, x)
					else:
						realx = alvo[0]
						realy = alvo[1]
						uav_move(realx, realy, x)	
						
					final_position = get_points([target_list[index]])[0]	
					distance = get_distance(realx,realy,final_position[0],final_position[1])
					#print(distance, [realx,realy], final_position)
					if(distance < 20):
						send_data(target_list[index], index, tick)
					
				tock = rospy.get_time()
				#print(x, str(2*distance/(tock-tick)))
			else:
				print('entrou')
				run_algo = True
				#sem.acquire()
				#print(auctioned_targets[xi])
				t1 = rospy.get_time()
				auctioned_targets[xi].remove(last_target)
				waiting_list.append((last_target, xi))
				#print(auctioned_targets[xi])
				#sem.release()
				
				
				
				

def main():

	#print(sys.argv[0])
	
	global targets
	global visits_table
	global time_table
	global auctioned_targets
	global waiting_list
	
	algo = int(sys.argv[1])
	handover = sys.argv[2].lower() == 'true'
	prediction = sys.argv[3].lower() == 'true'
	push = sys.argv[4].lower() == 'true'
	time = int(sys.argv[5])
	number_of_targets = int(sys.argv[6])
	spawn = sys.argv[7].lower()
	
	
	#print(algo, handover, prediction, push)

	
	uav_dict = dict(zip(uavs, get_points(uavs)))
	#targets_dict = dict(zip(targets, get_points([targets])))
	
	global world_size
	target_max_speed = 2
	target_z = 0
	uav_max_speed = 20
	uav_z = 20
	
	try:
		reset(0, boxes, uavs, targets, world_size)
		mov_id = _thread.start_new_thread(targets_movement, (targets, target_max_speed,world_size,target_z, push))
		#_thread.start_new_thread(monitor_distances, (targets, uavs,ts))
	except:
		print('Environment Error')

	targets = targets[:number_of_targets]
	global auctioned_targets 
	
	auctioned_targets = first_auction(auctioned_targets)
	while (len(auctioned_targets[0]) <= 1 or len(auctioned_targets[1]) <= 1 or len(auctioned_targets[2]) <= 1 or  len(auctioned_targets[3]) <= 1 or len(auctioned_targets[4]) <= 1):
		
		reset(0, boxes, uavs, targets, world_size)
		auctioned_targets = [[],[],[],[],[]]
		auctioned_targets = first_auction(auctioned_targets)
	
	
	
	if handover == True:
		_thread.start_new_thread(general_auction, ())
	
	print('Launching UAVS')
	
	tick = rospy.get_time()
	threads = []
	for xi, x in enumerate(uavs):
		
		try:
			#fly(xi, x, algo, tick, handover, prediction)
			threads.append(_thread.start_new_thread(fly, (xi, x, algo, tick, handover, prediction)))
		except:
			print('Flight Over!')
		
	extra_target = number_of_targets
	
	if(spawn = 'a'):
		for each in range(5):	
			rospy.sleep(time/15)
			sem.acquire()
			waiting_list.append([targets[extra_target], 6])
			sem.release()
			extra_target = extra_target + 1
		rospy.sleep(2 * time/ 3)
		
	elif(spawn = 'b'):
		rospy.sleep(time/6)
		for each in range(5):
			sem.acquire()
			waiting_list.append([targets[extra_target], 6])
			sem.release()
			extra_target = extra_target + 1
		rospy.sleep(5 * time /6)
		
	elif(spawn = 'c'):
		rospy.sleep(time)
		
	elif(spawn = 'd'):
		for each in range(10):	
			rospy.sleep(time/30)
			sem.acquire()
			waiting_list.append([targets[extra_target], 6])
			sem.release()
			extra_target = extra_target + 1
		rospy.sleep(2 * time/3)	
			
	elif(spawn = 'e'):
		rospy.sleep(time/6)
		for each in range(10):
			sem.acquire()
			waiting_list.append([targets[extra_target], 6])
			sem.release()
			extra_target = extra_target + 1
		rospy.sleep(5 * time /6)	
		
	elif(spawn = 'f'):
		rospy.sleep(time)
	
		
	print('this is the end')
	
	
	
	
	visitas = 0
	tempo = 0
	
	final_visits = []
	final_time = []
	for index, each in enumerate(results):
		visitas = len(each)
		if visitas > 1:
			tempo = (each[-1])/visitas
		elif visitas == 0:
			tempo = 0
		else:
			tempo = each[0]
		final_visits.append(visitas)
		final_time.append(tempo)
	
	params = ['Algo', 'handover', 'prediction', 'push', 'time',  'number_of_targets', 'spawn'] + targets
	descr_v = [algo, str(handover), str(prediction), str(push), time, number_of_targets, spawn] + final_visits
	descr_t = [algo, str(handover), str(prediction), str(push), time, number_of_targets, spawn] + final_time
	
	final_visits_dict = dict(zip(params, descr_v))
	final_time_dict = dict(zip(params, descr_t))
	visits_table = visits_table.append(final_visits_dict, ignore_index = True)
	time_table = time_table.append(final_time_dict, ignore_index = True)
	
	print(visits_table)
	
	visits_table.to_csv('./results/T4/visits_' + str(ts) + '_.csv', index = False, mode = 'a', header = False)
	time_table.to_csv('./results/T4/time_' + str(ts) +'_.csv', index = False, mode = 'a', header = False)
	print('Test Over')
		
	


if __name__ == '__main__':
	main()
	
	'''
    try:
        
    except:
        print("Basic Main Error")
	'''
       
