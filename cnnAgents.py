'''
width=20
height=11

import tensorflow as tf

WALL=1
DOT=2
BALL=3
ENEMY=4
GHOST=5
SELF=6

items=6

x=tf.placeholder(tf.float32,shape=[None,width*height])
y_=tf.placeholder(tf.float32,shape=[None,2])
W=tf.Variable(tf.zeros([width*height,2]))
b=tf.Variable(tf.zeros([2]))

y=tf.matmul(x,W)+b
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
'''

import game
import random
import util

def closestFood(pos,food,walls):
    fringe=[(pos[0],pos[1],0)]
    expanded=set()
    while fringe:
        pos_x,pos_y,dist=fringe.pop(0)
        if(pos_x,pos_y) in expanded:
            continue
        expanded.add((pos_x,pos_y))
        if food[pos_x][pos_y]:
            return dist
        nbrs=game.Actions.getLegalNeighbors((pos_x,pos_y),walls)
        for nbr_x,nbr_y in nbrs:
            fringe.append((nbr_x,nbr_y,dist+1))
    return None

def getFeatures(state,action):
    food=state.getFood()
    walls=state.getWalls()
    ghosts=state.getGhostPositions()
    #capsules=state.getCapsules()
    x,y=state.getPacmanPosition()
    dx,dy=game.Actions.directionToVector(action)
    nx,ny=int(x+dx),int(y+dy)
    features=dict()
    features["bias"]=1.0
    features["eatfood"] = 0.0
    features["closestfood"] = 0.0
    features["stepghosts"]=sum((nx,ny) in game.Actions.getLegalNeighbors(g,walls) for g in ghosts)
    if features["stepghosts"]==0 and food[nx][ny]:
        features["eatfood"]=1.0
    dist=closestFood((nx,ny),food,walls)
    if dist is not None:
        features["closestfood"]=float(dist)/(walls.width*walls.height)
    for k in features.keys():
        features[k]/=10.0
    return features

class QLAgent(game.Agent):
    def __init__(self,index=0):
        super().__init__(index)
        self.epsilon=0.05
        self.alpha=1.0
        self.discount=0.8
        self.weights=dict()
        self.lastState=None
        self.lastAction=None

    # Q(s,a) <-- Q(s,a) + alpha(r + gamma max(Q(s',a') - Q(s,a))
    def observationFunction(self,state):
        if self.lastState is None:
            return state
        reward=state.getScore()-self.lastState.getScore()
        features = getFeatures(state,self.lastAction)
        for key in features.keys():
            if self.weights.get(key) is None:
                self.weights[key]=0.0
            self.weights[key]+=self.alpha*(reward+self.discount*self.getValue(state)-self.getQValue(self.lastState,self.lastAction))
        return state

    def getAction(self,state):
        action=game.Directions.STOP
        legal = state.getLegalActions(self.index)
        if len(legal)>0:
            if util.flipCoin(self.epsilon):
                action=random.choice(legal)
            else:
                action=self.getPolicy(state,legal)
        self.lastState=state
        self.lastAction=action
        return action

    def getPolicy(self,state,legal):
        possibleQValues=dict()
        for action in legal:
            possibleQValues[action]=self.getQValue(state,action)
        if sum(possibleQValues.values())==0:
            return random.choice(legal)
        maxValue=-1
        maxKey=None
        for (k,v) in possibleQValues.items():
            if v>maxValue:
                maxValue=v
                maxKey=k
        return maxKey

    def getQValue(self,state,action):
        qValue=0.0
        features=getFeatures(state,action)
        for key in features.keys():
            if self.weights.get(key) is not None:
                qValue+=self.weights[key]*features[key]
        return qValue

    def getValue(self,state):
        qValues=dict()
        for action in state.getLegalActions(self.index):
            qValues[action]=self.getQValue(state,action)
        maxValue=-1
        for v in qValues.values():
            if v>maxValue:
                maxValue=v
        return maxValue

    def final(self,state):
        pass
