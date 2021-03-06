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

        self.gamma = 0.4
        self.epsilon = 0.1
        self.alpha = 0.001

        self.iter = 1
        self.sizea = 465+485
        self.sizeb = 70+400
        self.sizec = 84

        self.counta = self.sizea/(a)
        self.countb = self.sizeb/(b)
        self.countc = self.sizec/(c)

        self.Q = np.zeros((a, b, c, 2))
        self.counts = np.zeros((a, b, c, 2))

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


        '''
        self.cur_action = self.last_action
        self.last_action = new_action
        self.last_state  = self.cur_state
        self.cur_state = new_state'''
        #print state

        self.cur_score = state['score']
        

        return self.cur_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        
        if self.last_action!=None and self.cur_action!=None:
            
            #self.alpha = min(.01/self.iter, 0.001)
            self.counts[self.last_state][self.last_action] += 1.
            alpha = 1./self.counts[self.last_state][self.last_action]
            
            delta = reward + self.gamma * np.max(self.Q[self.cur_state]) - self.Q[self.last_state][self.last_action]

            self.Q[self.last_state][self.last_action] += alpha * delta
            

        self.last_reward = reward




iters = 200
trial = 2
group = [[5,5,3], [10, 10, 3], [10, 10, 5], [5, 5, 5]]
gammas = [0.3, 0.4, 0.5, 0.6]
record = np.zeros([len(group), len(gammas), trial, iters])
for i in xrange(len(group)):
    for k in xrange(len(gammas)):
        for j in xrange(trial):
            learner = Learner(10,10,3)
            learner.gamma = gammas[i]
            learner.epsilon = 0.0
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

                learner.iter = cur_iter+1


                # Loop until you hit something.
                while swing.game_loop():
                    pass

                cur_iter= learner.iter
                record[i][k][j][ii] = learner.cur_score

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
            print learner.gamma
            #pickle.dump(learner, open("learn_small.p", "wb"))
            print i, k, j, "Finished!"

pickle.dump(record, open("record_gamma.p", "wb"))



    
