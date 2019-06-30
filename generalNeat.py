'''
    neat-python :
    Title = neat-python,
    Author = Alan McIntyre and Matt Kallada and Cesar G. Miguel and Carolina Feher da Silva},
    https://github.com/CodeReclaimers/neat-python   
  
'''
#This script shows how the selected agent (agentName) plays the select game (gameName) 
    #Selected agent in this example :winner-winner_chicken_dinner295.pkl

#implement the NEAT algorith + python multiprocessing

import neat
import gym
from multiprocessing import Pool
import time
import pickle

# number of cores used in the pc
NUM_CORES = 5
gameName='DemonAttack-ram-v0'

#checkpoint values
#use_safe= False 
safe_number= 95
safe_range= 10

# eval_genomes values
min_fitness =10
check_interval=2000


'''
each game has a method to trasnlate the output 
to something that makes sense in the game
for example: 
             every left  movement changes to 3  
             every right movement changes to 2 

#0 up 
#1 button
#2 right 
#3 left 
#4 right + button
#5 left + button
#6 up + button
'''
def get_invaders_output(list):
    index=0
    for i in range (1,len(list)-1):
        if (list[i]>list[index]): index=i
        if index==6 : index=0
    return index  

def get_assault_output(list):
    index=0
    for i in range (1,len(list)-1):
        if (list[i]>list[index]): index=i
    value= index +1
    if  value==7 : value= 0
    return index  

def play_game(net):
    score=0
    env = gym.make(gameName)
    observation= env.reset()
    while True  :
        env.render()
        action= get_invaders_output(net.activate(observation))
        observation, reward, done, info = env.step(action)
        score+=reward
        if done:
            print "your final score: ", score
            break
    env.close


'''
runs the simulation of the agent in the game
 fitness = sum of the rewards its get 
 
 it stops the simulation before the game over, ifthe agent gets no reward (min_fitness) during many iterations (check_interval) . 
 this is made in order to avoid long simulation where the agent does not die, but won-t get more points either  
'''
def test_game(nets):
    #starting values 
    fitness=0
    move_buffer =[0,0,0,0]
    net = neat.nn.FeedForwardNetwork.create(nets[0], nets[1])
    #gym env 
    env = gym.make(gameName)
    observation= env.reset()
    
    counter=0
    temp_fitness=0
    while True :
        action= get_invaders_output(net.activate(observation))

        observation, reward, done, info = env.step(action)
        fitness+=reward
        counter +=1 
        temp_fitness+=reward
        if (counter==check_interval):
            if (temp_fitness<min_fitness):
                break
            else :
                temp_fitness=0
                counter=0
        if done  :
            break


    env.close
    print "fitness", fitness
    return fitness



'''
eval genos: gives its fitness to every genome in 'genomes'

    We get every fitness by using the multi processing with the 'test_game' def
        fitnesses=pool.map(test_game , nets)
    so the calculation of the fitnesses can be done in parellel
    
    we use the 'neats' to hold each array of (genome,config) 
    so it can be send to the send to the processing pool

'''
def eval_genomes(genomes, config):
    nets=[]
    for genome_id, tgenome in genomes:
        #net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append((tgenome,config))
    pool= Pool(NUM_CORES)
    fitnesses=pool.map(test_game , nets)
    pool.close()
    pool.join()
    cont=0 
    for genome_id, genome in genomes:
        genome.fitness=fitnesses[cont]
        cont+=1




# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-atari-pro')

if (safe_number>0):
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-'+str(safe_number))
else :
    p = neat.Population(config)



p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

if (safe_range>0):
    p.add_reporter(neat.Checkpointer(safe_range))

# Run until a solution is found.
t1=time.time()
winner = p.run(eval_genomes,105)
t2=time.time()
print "run time: ",t2-t1
stats.save_species_count(delimiter=',', filename='speciation.csv')
stats.save_genome_fitness(delimiter=',', filename='fitness_history.csv', with_cross_validation=False)


#saves the winner
with open('winner-winner_chicken_dinner'+str(safe_number)+'.pkl', 'wb') as output:
    pickle.dump(winner, output, pickle.HIGHEST_PROTOCOL)
# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
play_game(winner_net)

