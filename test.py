import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
from sko.GA import GA_TSP
from sko.ACA import ACA_TSP
from sko.SA import SA_TSP
from sko.IA import IA_TSP
from sko.demo_func import function_for_TSP
import pandas as pd
import time
import os

num_points = 25

points_coordinate = np.random.randint(low = -250, high = 250, size = (num_points,2))  # generate coordinate of points
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

genalg_df = pd.DataFrame(columns = ['size_pop', 'max_iter', 'prob_mut', 'best_distance', 'time'])
ant_df = pd.DataFrame(columns = ['size_pop', 'max_iter', 'best_distance', 'time'])
sm_df = pd.DataFrame(columns = ['T_max', 'T_min', 'L', 'best_distance', 'time'])
im_df = pd.DataFrame(columns = ['size_pop', 'max_iter', 'prob_mut', 'T', 'alpha',  'best_distance', 'time'])


genalg_df.to_csv("./results/0_genalg.csv", index= False)
ant_df.to_csv("./results/0_ant.csv", index= False)
sm_df.to_csv("./results/0_sm.csv", index= False)
im_df.to_csv("./results/0_im.csv", index= False)



def cal_total_distance(routine):

    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

def genalg(genalg_df, size_pop = 50, max_iter = 500, prob_mut = 1):

	begin = time.time()
	ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=size_pop, max_iter=max_iter, prob_mut = prob_mut)
	best_points, best_distance = ga_tsp.run()
	end = time.time()
	run = end - begin
	
	new_row = {'size_pop': size_pop, 'max_iter':max_iter, 'prob_mut':prob_mut, 'best_distance':best_distance[0], 'time':run}
	genalg_df = genalg_df.append(new_row, ignore_index=True)
	genalg_df.to_csv("./results/0_genalg.csv", mode='a', header=False, index= False)

	
def antcolony(ant_df, size_pop=50, max_iter=200):

	begin = time.time()
	aca = ACA_TSP(func=cal_total_distance, n_dim = num_points, size_pop = size_pop, max_iter = max_iter, distance_matrix=distance_matrix)
	best_x, best_y = aca.run()
	end = time.time()
	run = end - begin
	
	new_row = {'size_pop': size_pop, 'max_iter':max_iter, 'best_distance':aca.best_y, 'time':run}
	ant_df = ant_df.append(new_row, ignore_index=True)
	ant_df.to_csv("./results/0_ant.csv", mode='a', header=False, index= False)
	
	
def sim_anealing(sm_df, T_max=100, T_min=1, L=10):

	begin = time.time()
	sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=T_max, T_min=T_min, L=L * num_points)
	best_points, best_distance = sa_tsp.run()
	end = time.time()
	run = end - begin	
	
	new_row = {'T_max': T_max, 'T_min':T_min, 'L': L, 'best_distance':best_distance, 'time':run}
	sm_df = sm_df.append(new_row, ignore_index=True)
	sm_df.to_csv("./results/0_sm.csv", mode='a', header=False, index= False)
	
	
	

def immune(im_df, size_pop=500, max_iter=800, prob_mut=0.2, T=0.7, alpha=0.95):

	begin = time.time()
	ia_tsp = IA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=size_pop, max_iter=max_iter, prob_mut=prob_mut, T=T, alpha=alpha)
	best_points, best_distance = ia_tsp.run()
	
	end = time.time()
	run = end - begin
	
	new_row = {'size_pop': size_pop, 'max_iter':max_iter, 'prob_mut': prob_mut, 'T':T, 'alpha': alpha, 'best_distance':best_distance[0], 'time':run}
	sm_df = im_df.append(new_row, ignore_index=True)
	sm_df.to_csv("./results/0_im.csv", mode='a', header=False, index= False)

	


for x in range(100):
	
	for size_pop in [6, 10, 26, 50, 100, 250, 500, 1000]:
		for max_iter in [5, 10, 25, 50, 100, 250, 500, 1000]:
			for prob_mut in [0.001, 0.0025, 0.01, 0.1, 0.25, 0.5, 1]:
				print("Genetic Algorithm", x)
				print("size_pop: ", size_pop)
				print("max_iter: ", max_iter)
				print("prob_mut: ", prob_mut)
				genalg(genalg_df,size_pop, max_iter, prob_mut)
				os.system('cls' if os.name == 'nt' else 'clear') 

	for size_pop in [6, 10, 26, 50, 100, 250, 500, 1000]:
		for max_iter in [5, 10, 25, 50, 100, 250, 500, 1000]:
			print("Ant Colony", x)
			print("size_pop: ", size_pop)
			print("max_iter: ", max_iter)
			antcolony(ant_df, size_pop, max_iter)
			os.system('cls' if os.name == 'nt' else 'clear')

	for T_max in [6, 10, 26, 50, 100, 250, 500, 1000]:
		for T_min in [1, -10, -25, -50, -100, -250, -500, -1000]:
			for L in [5, 10, 25, 50, 100, 250, 500, 1000]:
				print("Simmuleate Annealing", x)
				print("T_max: ", T_max)
				print("T_min: ", T_min)
				print("L: ", L)
				sim_anealing(sm_df, T_max, T_min, L)
				os.system('cls' if os.name == 'nt' else 'clear')
				
	for size_pop in [6, 10, 26, 50, 100, 250, 500, 1000]:
		for max_iter in [5, 10, 25, 50, 100, 250, 500, 1000]:
			for prob_mut in [0.001, 0.0025, 0.01, 0.1, 0.25, 0.5, 0.75, 1]:
				for T in [0.001, 0.0025, 0.01, 0.1, 0.25, 0.5, 0.75, 1]:
					for alpha in [0.001, 0.0025, 0.01, 0.1, 0.25, 0.5, 1]:
						print("Immune Algorithm", x)
						print("size_pop: ", size_pop)
						print("max_iter: ", max_iter)
						print("prob_mut: ", prob_mut)
						print("T: ", T)
						print("alpha: ", alpha)
						immune(im_df, size_pop, max_iter, prob_mut, T, alpha)
						os.system('cls' if os.name == 'nt' else 'clear') 

