# Recycling NEAT populations in Atari Games
## Ramificacion-Neuro-evolutiva 


The idea of this project is to generate a population of agents that is able to solve a game using the **NEAT (Neuroevolution of augmenting topologies)** algorithm, then use this population in a similar but different environment(game) and try to continue the evolution process in this new environment. 

This experiment was done using:
Open AI gym to genereate the Atari enviroments 
[OPENAI GYM](https://github.com/openai/gym)

NEAT-pthon as the implemantation of NEAT to create and evolve the neural nets 
[NEAT-PYTHON](https://neat-python.readthedocs.io/en/latest/)

For serializing and de-serializing object structure, in this case the agents(NEAT-python also use it internally).
[Pickle](https://docs.python.org/3/library/pickle.html/)



Basically I wanted to replicate what happens in nature, when you have a species or a group of species, and the environment they live in suffer a change(climate change for example). Then let those species evolve in this new environment over time. This project tries to emulate this process by using video games as the environments, and neural networks as the individuals of the species, where scoring enough points means survival (fitnes). 

For this experiment I used Atari shot-them-up games. I tried that the first game was the most simple one, and then evolved the remaining population in two different games, with a little bit more of complexity. This was made in order to emulate the evolution of the species when new elements are incorporated into the environment. 

This experiment was made using the NEAT (Neuroevolution of augmenting topologies) algorithm, this is a genetic algorithm that evolves a population of neural networks, dividing them into species. NEAT has many unique characteristics but what makes it so special, is how it accomplishes to do the crossover of the neural nets, mixing the topologies of both networks in a significant way,  using historical markers

Proyect presentaion: https://www.canva.com/design/DADxgGs4ces/phPG_6drAxn70WQU4e5p2g/view?utm_content=DADxgGs4ces&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink

