import numpy.random as npr
import numpy as np
import sys
import cPickle as pickle
import math

from SwingyMonkey import SwingyMonkey



class Learner:

    def __init__(self, a=38, b=121, c=27):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.cur_state = None
        self.cur_action = None
        self.cur_score = 0
        self.rel = None
        self.dis = None
        self.vel = None

        self.gamma = 0.4
        self.epsilon = 0.1
        self.alpha = 0.001

        self.iter = 1
        self.sizea = 485+115
        self.sizeb = 164+351
        self.sizec = 43+28

        self.counta = self.sizea/(a)
        self.countb = self.sizeb/(b)
        self.countc = self.sizec/(c)

        self.Q = np.zeros((a, b, c, 2))
        #self.Q = np.zeros((10, 20, 10, 2))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.cur_state = None
        self.cur_action = None
        self.cur_score = 0
        self.iter +=1


    def state_to_number(self, state):
        a = (485-state['tree']['dist'])/self.counta

        b = (70 - state['tree']['bot']+state['monkey']['bot'])/self.countb

        c = (state['monkey']['vel']+39)/self.countc

        if self.rel == None:
            self.rel = [state['tree']['bot']-state['monkey']['bot'], state['tree']['bot']-state['monkey']['bot']]
        else:
            self.rel = [min(state['tree']['bot']-state['monkey']['bot'], self.rel[0]), max(state['tree']['bot']-state['monkey']['bot'], self.rel[1])]

        if self.vel == None:
            self.vel = [state['monkey']['vel'], state['monkey']['vel']]
        else:
            self.vel = [min(state['monkey']['vel'], self.vel[0]), max(state['monkey']['vel'], self.vel[1])]

        if self.dis == None:
            self.dis = [state['tree']['dist'], state['tree']['dist']]
        else:
            self.dis = [min(state['tree']['dist'], self.dis[0]), max(state['tree']['dist'], self.dis[1])]


        #print state['tree']['dist'], state['tree']['bot']-state['monkey']['bot'], state['monkey']['vel']

        return (a,b,c)

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.

        new_state  =  self.state_to_number(state)

        new_action = np.argmax(self.Q[new_state])

        if npr.rand() < self.epsilon:
            new_action = npr.rand()<0.1

        
        self.last_action = self.cur_action
        self.cur_action = new_action
        self.last_state  = self.cur_state
        self.cur_state = new_state
        
        #print state['tree']['bot']-state['monkey']['bot'],
        #print state['tree']['dist'],
        #print state['monkey']['vel']
        #print new_state


        '''
        self.cur_action = self.last_action
        self.last_action = new_action
        self.last_state  = self.cur_state
        self.cur_state = new_state'''
        #print state

        self.cur_score = state['score']
        

        return False#self.cur_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        
        if self.last_action!=None and self.cur_action!=None:
            
            #self.alpha = min(.01/self.iter, 0.001)
            
            delta = reward + self.gamma * np.max(self.Q[self.cur_state]) - self.Q[self.last_state][self.last_action]

            self.Q[self.last_state][self.last_action] += self.alpha * delta
            

        self.last_reward = reward

rel = None
vel = None
dis = None


iters = 50
trial = 1
group = [[5,5,3], [10, 10, 3], [10, 10, 5], [5, 5, 5]]
record = np.zeros([len(group), trial, iters])
for i in xrange(1):
    for j in xrange(trial):
        learner = Learner(*group[i])
        #learner = pickle.load(open("learn_small.p", "rb"))
        learner.gamma = 0.4
        learner.alpha = 0.001
        learner.epsilon = 0.1
        score = 0
        cur_iter = 0
        scores = []

        for ii in xrange(iters):

            # Make a new monkey object.
            swing = SwingyMonkey(sound=False,            # Don't play sounds.
                                 text="Epoch %d" % (ii), # Display the epoch on screen.
                                 tick_length=1,          # Make game ticks super fast.
                                 render=False,
                                 action_callback=learner.action_callback,
                                 reward_callback=learner.reward_callback)

            #learner.epsilon = 0.001
            #learner.alpha = min(.0001, 1./(ii+1.))
            #learner.epsilon = 1./10000.
            learner.iter = cur_iter+1
            #learner.alpha = min(1./(ii+1.), 0.05)


            # Loop until you hit something.
            while swing.game_loop():
                pass

            cur_iter= learner.iter
            record[i][j][ii] = learner.cur_score

            score = max(score, learner.cur_score)
            scores.append(learner.cur_score)
            if not ii%20:
                #pickle.dump(learner, open("learner_record.p", "wb"))
                print "Iteration:",ii, "Highest Score:", score, "This Score:", learner.cur_score,"Avg:", np.mean(scores)#, learner.alpha, learner.iter
            #print learner.iter, learner.gamma, learner.alpha, learner.epsilon

            # Reset the state of the learner.
            learner.reset()

        print "Iteration:",ii, "Highest Score:", score, "This Score:", learner.cur_score,"Avg:", np.mean(scores)
        print np.sum(learner.Q>0)
        print learner.alpha
        #pickle.dump(learner, open("learn_small.p", "wb"))
        print i, "Finished!"

print "relative postion:", learner.rel
print "distance", learner.dis
print "velocity", learner.vel



    
