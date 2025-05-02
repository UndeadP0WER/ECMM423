import numpy as np
import matplotlib.pyplot as plt
import random
import lstm
from lstm import generate_dataset, LSTMParams, LSTMModel

import pickle

random.seed()

def main():
    #setupData(30)
    
    
    params = LSTMParams.from_arr([0.307777813226747, 0.3904958476912733, 0.9626740542136294, 0.45931241527424915, 0.6354211965871436])
    print("evaluating params: ", params.hidden_size, params.num_layers, params.dropout, params.lr, params.epochs)
    
    #return
    
    iters = 15
    
    params = PSOParams(5, 0.7, 1.8, 1.8)
    
    
    #pso
    #print("PSO")
    #b, bst, pst = particleSwarmOptimisation(5, params, iters, lstm_fit)
    #print("best found weights:", b)
    #
    ##saves data to file
    #obj = (b, bst, pst)
    #with open('data/PSO_DATA2.pkl', 'wb') as output:
    #    pickle.dump(obj, output)
    #
    #
    #with open('data/PSO_DATA2.pkl', 'rb') as pkl_file:
    #    data = pickle.load(pkl_file)
    #b=data[0]
    #bst=data[1]
    #pst=data[2]
    #
    #
    #plt.figure(figsize=(10, 6))
    #scx, scy = flatten_w_index(pst)
    #plt.yscale('log')
    #plt.scatter(scx, scy)
    #plt.plot(np.arange(0,len(bst)), bst)
    #
    #lasty = ""
    #for x,y in zip(np.arange(0,len(bst)), bst):
    #    if y != lasty:
    #        label = "{:.2f}".format(y)
    #        lasty = y
    #    else:
    #        label = ""
    #    plt.annotate(label, # this is the text
    #                (x,y), # these are the coordinates to position the label
    #                textcoords="offset points", # how to position the text
    #                xytext=(0,10), # distance from text to points (x,y)
    #                ha='center') # horizontal alignment can be left, right or center
    #
    #
    #plt.title('PSO Fitness By Generation')
    #plt.xlabel('Generation')
    #plt.ylabel('Fitness')
    #plt.savefig("img/pso_2.png")
    #plt.show()
    #print(min(bst))
    
    
    #random search
    #print("Random Search")
    #b, bst, pst = randomSearch(5, iters * params.pop, lstm_fit)
    #print("best found weights:", b)
    #
    ##saves data to file
    #obj = (b, bst, pst)
    #with open('data/RND_DATA2.pkl', 'wb') as output:
    #    pickle.dump(obj, output)
    #
    #with open('data/RND_DATA2.pkl', 'rb') as pkl_file:
    #    data = pickle.load(pkl_file)
    #b=data[0]
    #bst=data[1]
    #pst=data[2]
    #
    #
    #plt.figure(figsize=(10, 6))
    #plt.scatter(np.arange(0,len(pst)), pst)
    #plt.plot(np.arange(0,len(bst)), bst)
    #plt.yscale('log')
    #
    #lasty = ""
    #for x,y in zip(np.arange(0,len(bst)), bst):
    #    if y != lasty:
    #        label = "{:.2f}".format(y)
    #        lasty = y
    #    else:
    #        label = ""
    #    plt.annotate(label, # this is the text
    #                (x,y), # these are the coordinates to position the label
    #                textcoords="offset points", # how to position the text
    #                xytext=(0,10), # distance from text to points (x,y)
    #                ha='center') # horizontal alignment can be left, right or center
    #
    #plt.title('Random Search Fitness')
    #plt.xlabel('Generation')
    #plt.ylabel('Fitness')
    #plt.savefig("img/rnd_2.png")
    #plt.show()
    #print(min(bst))
    
    
    #stochastic hill climb
    #print("Stochastic Hill Climber")
    #b, bst, pst = hillClimb(5, iters * params.pop, lstm_fit)
    #print("best found weights:", b)
    #
    ##saves data to file
    #obj = (b, bst, pst)
    #with open('data/SHC_DATA2.pkl', 'wb') as output:
    #    pickle.dump(obj, output)
    #  
    #with open('data/SHC_DATA2.pkl', 'rb') as pkl_file:
    #    data = pickle.load(pkl_file)
    #b=data[0]
    #bst=data[1]
    #pst=data[2]
    #
    #plt.figure(figsize=(10, 6))
    #plt.scatter(np.arange(0,len(pst)), pst)
    #plt.plot(np.arange(0,len(bst)), bst)
    #plt.yscale('log')
    #
    #lasty = ""
    #for x,y in zip(np.arange(0,len(bst)), bst):
    #    if y != lasty:
    #        label = "{:.2f}".format(y)
    #        lasty = y
    #    else:
    #        label = ""
    #    plt.annotate(label, # this is the text
    #                (x,y), # these are the coordinates to position the label
    #                textcoords="offset points", # how to position the text
    #                xytext=(0,10), # distance from text to points (x,y)
    #                ha='center') # horizontal alignment can be left, right or center
    #
    #plt.title('Stochastic Hill Climber Fitness')
    #plt.xlabel('Generation')
    #plt.ylabel('Fitness')
    #plt.savefig("img/shc_2.png")
    #plt.show()
    #print(min(bst))
    
    


#takes a 2d array (like whats produced in PSO)
#and flattens it to a 1d array
#and produces an array with the first index of the element
# eg an input of [[1,2,3],[4,5,6]] would produce
# [0,0,0,1,1,1], [1,2,3,4,5,6]
def flatten_w_index(arr_2d):
    y = []
    x = []
    for i in range(0, len(arr_2d)):
        for j in range(0, len(arr_2d[i])):
            x.append(i)
            y.append(arr_2d[i][j])
    return x, y

#generates a dataset and then plots it (and saves it)
def setupData(year_count):
    # Plot the synthetic data
    df = generate_dataset(year_count)
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Value'], label='Synthetic Data')
    plt.title('Synthetic Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig("img/dataset.png")
    plt.show()


#calculates the mean squared error between two arrays
def eval(predicted_arr, actual_arr):
    sum = 0
    for i in range(0, len(predicted_arr)):
        sum += (actual_arr[i] - predicted_arr[i])**2
    
    sum = sum/len(predicted_arr)
    return sum

#calculates the suitability of a set of hyperparameters by training an LSTM model and evaluating its effectiveness
def lstm_fit(arr):
    params = LSTMParams.from_arr(arr)
    print("evaluating params: ", params.hidden_size, params.num_layers, params.dropout, params.lr, params.epochs)
    model = LSTMModel(output_size=1, params=params)    
    tp, ta = model.train_model()
    e = eval(tp, ta)
    print(e)
    return e

#gives a random output. useful for testing
def rndfit(arr):
    return random.random()*255


#Particle Swarm Optimisation
class Particle:
    #best = []
    #cur = []
    #vel = []
    def __init__(self, arr_len):
        #generates a starting position 
        #uses float because PSOs operate best on continuous solution space
        #so a binary string is achieved by rounding later on
        #distributed by xi ~ U(blo, bup) (uniform between 0,1)
        self.cur = []
        for i in range(0, arr_len):
            self.cur.append(random.random())
        #best known position is set to starting position 
        self.best = self.cur.copy()
        #generates velocity
        #distributed by vi ~ U(-|bup-blo|, |bup-blo|) (uniform between -1,1)
        self.vel = []
        for i in range(0, arr_len):
            self.vel.append((random.random()*2)-1)
        
        self.bestfit = 0;
    
    def __str__(self):
        #mostly for testing
        return "best:" + str(self.best) + ",\ncur: " + str(self.cur) + ",\nvel: " + str(self.vel)

class PSOParams:
    def __init__(self, popsize, inertia_weight, cognitive_coeff, social_coeff):
        self.pop = popsize
        self.w = inertia_weight
        self.cc = cognitive_coeff
        self.sc = social_coeff
    
def particleSwarmOptimisation(arr_len, params, iterations, fitfunct):
    #values for graphing
    gen_fits = [[]]
    gen_best = []

    #init particles distributed uniformly around the solution space
    pop = []
    for p in range(0, params.pop):
        pop.append(Particle(arr_len));
    
    #sets swarms best known positions
    gbest = pop[0].best.copy()
    gbestfit = fitfunct(gbest)
    gen_best = [gbestfit] #just in case the best is the first one, as it doesnt get updated later if it is
    
    for p in pop:
        
        newfit = fitfunct(p.best)
        #sets current best fit for each particle
        p.bestfit = newfit
        
        #notes current fit for the particle in the generation tracker
        gen_fits[0].append(newfit)
        
        #if current particle's best is better than the previous global best, replace global best
        if newfit < gbestfit:
            gbest = p.best.copy()
            gbestfit = newfit
            gen_best[0] = newfit
    
    
    for iter in range(0, iterations):
        iter_best = gen_best[-1] #sets the iteration best to the previous iterations best - best line can only better solutions
        iter_fits = []
        
        print("Iteration", iter, "of", iterations)
        
        for p in pop:
            rp = random.random()
            rg = random.random()
            for i in range(0,arr_len):
                #updates velocity based on: vi,d ← w vi,d + φp rp (pi,d-xi,d) + φg rg (gd-xi,d)
                p.vel[i] = (params.w * p.vel[i]) + (params.cc * rp * (p.best[i] - p.cur[i])) + (params.sc * rg * (gbest[i] - p.cur[i]))
    
                #updates position
                p.cur[i] = p.cur[i] + p.vel[i]
                #clamps back in range
                if p.cur[i] > 1:
                    p.cur[i] = 1
                elif p.cur[i] < 0:
                    p.cur[i] = 0
            
            #checks fitness function
            fit = fitfunct(p.cur)
            iter_fits.append(fit)
            
            if fit < p.bestfit:
                #new particle best
                p.bestfit = fit
                p.best = p.cur.copy()
                
                if fit < gbestfit:
                    #new global best
                    gbestfit = fit
                    gbest = p.cur.copy()
                    iter_best = fit
        
        #updates tracking values at the end of the iteration
        gen_best.append(iter_best)
        gen_fits.append(iter_fits)

    return gbest, np.array(gen_best), np.array(gen_fits)

#Random Search
def randomSearch(arr_len, iterations, fitfunct):
    #generate initial solution
    best = []
    bestfit = 0
    
    #graph trackers for analysisng the random search
    all_fits = [] 
    best_fits = [] 
    
    for i in range(0, arr_len):
        best.append(random.random())
        pass
    bestfit = fitfunct(best)
    all_fits.append(bestfit)
    best_fits.append(bestfit)
    
    #iterates over the random search for the defined number of iterations
    for i in range(0, iterations):
        
        print("Iteration", i, "of", iterations)

        #generates a competitior solution
        new = []
        for i in range(0, arr_len):
            new.append(random.random())
        #calcs new fitness
        newfit = fitfunct(new)
        #adds to graph tracker
        all_fits.append(newfit)
        
        #if new solution is a better fit, replace best fit
        if newfit < bestfit:
            best = new
            bestfit = newfit
        
        best_fits.append(bestfit) #keeps track of the best known at each iteration
        
    return best, best_fits, all_fits

#Stochastic Hill Climbing
## picks a random neighbouring solution to compare
def hillClimb(arr_len, iterations, fitfunct):
    #generate initial solution
    best = []
    bestfit = 0
    
    #graph trackers for analysisng the random search
    all_fits = [] 
    best_fits = [] 
    
    for i in range(0, arr_len):
        best.append(random.random())
        pass
    bestfit = fitfunct(best)
    
    all_fits.append(bestfit)
    best_fits.append(bestfit)
    
    #iterates over the random search for the defined number of iterations
    for i in range(0, iterations):
    
        print("Iteration", i, "of", iterations)
        
        print("best", best, bestfit)
        #generates a competitior solution
        new = best.copy()
        #picks a random position, replaces it with a random selection
        new[int(random.random()*arr_len)] = random.random()
        
        #calcs new fitness
        newfit = fitfunct(new)
        #adds to graph tracker
        all_fits.append(newfit)
        
        #if new solution is a better fit, replace best fit
        if newfit < bestfit:
            best = new.copy()
            bestfit = newfit
        
        best_fits.append(bestfit) #keeps track of the best known at each iteration
    
    return best, best_fits, all_fits



#main
if __name__=="__main__":
    main()