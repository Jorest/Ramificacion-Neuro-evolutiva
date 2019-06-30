'''
    neat-python :
    Title = neat-python,
    Author = Alan McIntyre and Matt Kallada and Cesar G. Miguel and Carolina Feher da Silva},
    https://github.com/CodeReclaimers/neat-python   
  
'''
#This script shows how the selected agent (agentName) plays the select game (gameName) 
    #Selected agent in this example :winner-winner_chicken_dinner295.pkl



import neat
import gym
from multiprocessing import Pool
import time
import pickle

gameName='Assault-ram-v0'
#gameName='DemonAttack-ram-v0'
#gameName='Assault-ram-v0'
#gameName='BeamRider-ram-v0'


agentName='winner-winner_chicken_dinner295.pkl'


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


def get_invaders_demon_output(list):
    index=0
    for i in range (1,len(list)-1):
        if (list[i]>list[index]): index=i
        if index==6 : index=0
    return index  

def get_assault_output(list):
    index=0
    value=0
    for i in range (1,len(list)-1):
        if (list[i]>list[index]): index=i

    value= index +1
    if  value==7 : value= 0
    return value  

def get_beam_output(list):
    index=0
    value=0
    for i in range (1,len(list)-1):
        if (list[i]>list[index]): index=i
    if (index==2 or index==3):
        value = index +1 
    if (index==4 or index==5):
        value = index +3
    if (index == 6) : 
        value=2

    return value  

#runs the observacion-Action cycle (plays the game)
def play_game(net):
    score=0
    env = gym.make(gameName)
    observation= env.reset()
    while True  :
        env.render()
        #0.015 era antes ahora sera menos 
        time.sleep(0.02) 
        action= get_assault_output(net.activate(observation))
        observation, reward, done, info = env.step(action)
        score+=reward
        if done:
            print "your final score: ", score
            break
    env.close
    




# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-atari-pro')





with open(agentName, 'rb') as input:
    winner = pickle.load(input)


# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
play_game(winner_net)

