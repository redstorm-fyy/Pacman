import tensorflow as tf
import numpy as np
import collections
import random
import game
import util
from learningAgents import ReinforcementAgent

class DeepQNet:
    def __del__(self):
        self.sess.close()

    def __init__(self,width,height,channels):
        self.ydim=4
        self.x=tf.placeholder(tf.float32,[None,width,height,channels])
        self.qvalue=tf.placeholder(tf.float32,[None])
        self.reward=tf.placeholder(tf.float32,[None])
        self.terminal=tf.placeholder(tf.float32,[None])
        self.actions=tf.placeholder(tf.float32,[None,self.ydim])

        size=3;filters=32
        weight=tf.Variable(tf.random_normal([size,size,channels,filters],stddev=0.1))
        bias=tf.Variable(tf.constant(0.1,shape=[filters]))
        conv=tf.nn.conv2d(self.x,weight,strides=[1,1,1,1],padding="VALID")+bias
        conv=tf.nn.relu(conv)
        layer=conv
        #layer=tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

        layerShape=layer.get_shape().as_list()
        indim=layerShape[1]*layerShape[2]*layerShape[3];outdim=256
        flat=tf.reshape(layer,[-1,indim])
        weight=tf.Variable(tf.random_normal([indim,outdim]))
        bias=tf.Variable(tf.constant(0.1,shape=[outdim]))
        layer=tf.nn.relu(tf.matmul(flat,weight)+bias)

        indim=outdim;outdim=self.ydim;
        weight=tf.Variable(tf.random_normal([indim,outdim],stddev=0.1))
        bias=tf.Variable(tf.constant(0.1,shape=[outdim]))
        self.y=tf.matmul(layer,weight)+bias

        discount=tf.constant(0.9)
        newQ=self.reward+(1.0-self.terminal)*discount*self.qvalue
        predictQ=tf.reduce_sum(self.y*self.actions,reduction_indices=1)
        loss=tf.reduce_sum(tf.pow(newQ-predictQ,2))

        optimizer=tf.train.RMSPropOptimizer(0.001,decay=0.99)
        self.rmsprop=optimizer.minimize(loss)

        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self,batchSize,state,actions,nextState,reward,terminal):
        qvalue=np.zeros(batchSize)
        feed_dict={self.x:nextState,self.qvalue:qvalue,self.actions:actions,self.terminal:terminal,self.reward:reward}
        qvalue=self.sess.run(self.y,feed_dict=feed_dict)
        qvalue=np.amax(qvalue,axis=1)
        feed_dict={self.x:state,self.qvalue:qvalue,self.actions:actions,self.terminal:terminal,self.reward:reward}
        self.sess.run(self.rmsprop,feed_dict=feed_dict)

    def values(self,state):
        feed_dict={self.x:state,self.qvalue:np.zeros(1),self.actions:np.zeros([1,self.ydim]),self.terminal:np.zeros(1),self.reward:np.zeros(1)}
        qvalue=self.sess.run(self.y,feed_dict=feed_dict)
        return qvalue[0]

GDQN=None

def makesureDQN(width,height,channels):
    global GDQN
    if GDQN is None:
        GDQN=DeepQNet(width,height,channels)
    return GDQN

def getDirection(index):
    if index==0.:
        return game.Directions.NORTH
    elif index==1.:
        return game.Directions.EAST
    elif index==2.:
        return game.Directions.SOUTH
    elif index==3.:
        return game.Directions.WEST
    else:
        return game.Directions.STOP

def translateAction(action):
    if action==game.Directions.NORTH:
        return [1,0,0,0]
    elif action==game.Directions.EAST:
        return [0,1,0,0]
    elif action==game.Directions.SOUTH:
        return [0,0,1,0]
    elif action==game.Directions.WEST:
        return [0,0,0,1]
    else:
        return [0,0,0,0]

def translateState(state):
    walls=state.getWalls()
    width=walls.width
    height=walls.height
    channels=6
    tensor=np.zeros([width,height,channels],dtype=int)
    food=state.getFood()
    for i in range(width):
        for j in range(height):
            vector=tensor[i][j]
            vector[0]=walls[i][j] #wall
            vector[1]=food[i][j]  #food
    for (x,y) in state.getCapsules():
        tensor[x][y][2]=1 #capsule
    for ghostState in state.getGhostStates():
        x,y=ghostState.getPosition()
        x,y=int(x),int(y)
        if ghostState.scaredTimer>0:
            tensor[x][y][3]=1 #scaredghost
        else:
            tensor[x][y][4]=1 #ghost
    x,y=state.getPacmanPosition()
    tensor[x][y][5]=1 #pacman
    return tensor

class DQNAgent(ReinforcementAgent):
    def __init__(self,**args):
        super().__init__(**args)
        self.replay=collections.deque()
        self.epsilon=0.05
        self.frameNum=100000
        self.batchSize=32
        self.startCount=1024
        self.count=0

    def registerInitialState(self,state):
        super().registerInitialState(state)
        walls = state.getWalls()
        width = walls.width
        height = walls.height
        channels = 6  # pacman,ghost,wall,food,capsule,scaredghost
        self.dqn = makesureDQN(width, height, channels)
        self.terminal=False

    def getPolicy(self,state,legalActions):
        state=translateState(state)
        state=state[np.newaxis]
        qvalue=self.dqn.values(state)

        actArray=[]
        for i,v in enumerate(qvalue):
            actArray.append((i,v))
        actArray.sort(key=lambda e:e[1])
        actLen=len(actArray)
        for (idx,act) in enumerate(reversed(actArray)):
            i=act[0]
            v=act[1]
            if getDirection(i) not in legalActions:
                del(actArray[actLen-idx-1])
            else:
                break

        idx=actArray[len(actArray)-1][0]
        action=getDirection(idx)
        if action not in legalActions:
            action=game.Directions.STOP
        return action

    def getAction(self,state):
        legalActions=self.getLegalActions(state)
        if len(legalActions)==0:
            return None
        action=None
        if util.flipCoin(self.epsilon):
            action=random.choice(legalActions)
        else:
            action=self.getPolicy(state,legalActions)
        self.doAction(state,action)
        return action

    def update(self,state,action,nextState,reward):
        if self.epsilon==0:
            return
        state=translateState(state)
        nextState=translateState(nextState)
        action=translateAction(action)
        frame=(state,action,nextState,reward,self.terminal)
        self.replay.append(frame)
        if len(self.replay)>self.frameNum:
            self.replay.popleft()
        self.count+=1
        #if self.count%self.batchSize!=0:
        #    return
        if self.count<self.startCount:
            return
        self.train()

    def train(self):
        batch=random.sample(self.replay,self.batchSize)
        state=[]
        action=[]
        nextState=[]
        reward=[]
        terminal=[]
        for frame in batch:
            state.append(frame[0])
            action.append(frame[1])
            nextState.append(frame[2])
            reward.append(frame[3])
            terminal.append(frame[4])
        self.dqn.train(self.batchSize,state,action,nextState,reward,terminal)

    def final(self,state):
        self.terminal=True
        super().final(state)

