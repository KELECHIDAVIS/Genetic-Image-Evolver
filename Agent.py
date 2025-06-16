import random
import numpy as np
from Circle import Circle


class Agent:
    # agents are given a previous genome of circles to mutate slightly
    #genome of circles is held within canvas where
    #if empty the agent is in the first generation so make a new circle


    #if this mutated agent has a better fitness than its parent it'll be included in next pop

    def __init__(self,  genome, canvas_size):
        #and new random circle
        self.genome = self.mutate(genome)
        self.canvas_size = canvas_size
        self.fitness = 10000000 #temp val

    # mutations include:
    # new random circle
    def mutate(self, genome):
        #TODO tinker with ranges of diff mutation types to see what works best
        color = (random.random(), random.random(), random.random(), random.random())
        pos = (random.randrange(0,self.canvas_size[0]), random.randrange(0,self.canvas_size[1]))
        radius = random.randrange(0,self.canvas_size[0])
        return genome+ [Circle(pos, radius, color)]
