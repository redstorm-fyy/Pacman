import tensorflow as tf
import numpy as np
import collections
import random
import game
import util
from learningAgents import ReinforcementAgent

class QFunction:

    def addConv2d(self,layer, size, channels, filters):
        weight = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[filters]))
        conv = tf.nn.conv2d(layer, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
        layer = tf.nn.relu(conv)
        self.var.append(weight)
        self.var.append(bias)
        tf.summary.histogram("weight", weight)
        tf.summary.histogram("bias", bias)
        return layer

    def addLayer(self,layer, indim, outdim, actfn=None):
        weight = tf.Variable(tf.random_normal([indim, outdim],stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[outdim]))
        layer = tf.matmul(layer, weight) + bias
        if actfn is not None:
            layer = actfn(layer)
        self.var.append(weight)
        self.var.append(bias)
        tf.summary.histogram("weight", weight)
        tf.summary.histogram("bias", bias)
        return layer,weight

    def __init__(self,x,channels,ydim,keep_prob):
        self.var=[]
        self.regular=None

        with tf.name_scope("conv"):
            layer = self.addConv2d(x, 4, channels, 64)

        with tf.name_scope("hidden"):
            layerShape = layer.get_shape().as_list()
            indim = layerShape[1] * layerShape[2] * layerShape[3];
            layer = tf.reshape(layer, [-1, indim])
            layer,self.regular = self.addLayer(layer, indim, 256, actfn=tf.nn.relu)
            layer = tf.nn.dropout(layer, keep_prob=keep_prob)

        with tf.name_scope("output"):
            self.y,_ = self.addLayer(layer, 256, ydim)


class DeepQNet:
    def __del__(self):
        self.sess.close()

    def __init__(self,width,height,channels,ydim,logdir=None,savedir=None):
        self.ydim=ydim
        self.qvalue=tf.placeholder(tf.float32,[None])
        self.reward=tf.placeholder(tf.float32,[None])
        self.actions=tf.placeholder(tf.float32,[None,self.ydim])
        self.keep_prob=tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32, [None, width, height, channels])

        self.qfunc=QFunction(self.x,channels,self.ydim,self.keep_prob)
        self.tfunc=QFunction(self.x,channels,self.ydim,1)

        with tf.name_scope("synchro"):
            self.sychro=[]
            for i,var in enumerate(self.qfunc.var):
                self.sychro.append(tf.assign(self.tfunc.var[i],var))

        with tf.name_scope("loss"):
            newQ=self.reward+0.9*self.qvalue
            predictQ=tf.reduce_sum(self.qfunc.y*self.actions,reduction_indices=1)
            loss=tf.reduce_sum(tf.square(newQ-predictQ)+0.01*tf.nn.l2_loss(self.qfunc.regular))
            tf.summary.scalar("loss",loss)
            tf.summary.histogram("qvalue",predictQ)

        with tf.name_scope("optimizer"):
            self.global_step=tf.Variable(0,trainable=False)
            optimizer=tf.train.RMSPropOptimizer(0.001,decay=0.99)
            self.rmsprop=optimizer.minimize(loss,global_step=self.global_step)

        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.logdir=logdir
        if self.logdir is not None:
            self.writer=tf.summary.FileWriter(self.logdir,self.sess.graph)
            self.summary=tf.summary.merge_all()
        self.savedir=savedir
        if self.savedir is not None:
            self.saver = tf.train.Saver()
            savefile=tf.train.latest_checkpoint(self.savedir)
            if savefile is not None:
                self.saver.restore(self.sess,savefile)

    def train(self,state,actions,reward,qvalue):
        feed_dict={self.x:state,self.actions:actions,self.reward:reward,self.qvalue:qvalue,self.keep_prob:1.0}#maybe 0.5
        _,step=self.sess.run([self.rmsprop,self.global_step],feed_dict=feed_dict)
        #step=tf.train.global_step(self.sess, self.global_step)
        if step%100==0:
            self.sess.run(self.sychro)
        if self.logdir is not None:
            if step%100==0:
                summary=self.sess.run(self.summary,feed_dict=feed_dict)
                self.writer.add_summary(summary,step)
        if self.savedir is not None:
            if step%1000==0:
                self.saver.save(self.sess, self.savedir, global_step=step)
        return step

    def max_qvalue(self,state):
        feed_dict={self.x:state}
        qvalue=self.sess.run(self.tfunc.y,feed_dict=feed_dict)
        qvalue=np.amax(qvalue,axis=1)
        return qvalue

    def qvalue_distribution(self,state):
        feed_dict={self.x:state,self.keep_prob:1.0}
        qvalue=self.sess.run(self.qfunc.y,feed_dict=feed_dict)
        return qvalue[0]

def getDirection(index):
    if index==0.:
        return game.Directions.NORTH
    elif index==1.:
        return game.Directions.EAST
    elif index==2.:
        return game.Directions.SOUTH
    elif index==3.:
        return game.Directions.WEST
    elif index==4.:
        return game.Directions.STOP

def translateAction(action):
    if action==game.Directions.NORTH:
        return [1,0,0,0,0]
    elif action==game.Directions.EAST:
        return [0,1,0,0,0]
    elif action==game.Directions.SOUTH:
        return [0,0,1,0,0]
    elif action==game.Directions.WEST:
        return [0,0,0,1,0]
    elif action==game.Directions.STOP:
        return [0,0,0,0,1]

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

    def final(self,state):
        super().final(state)

    def __init__(self,**args):
        super().__init__(**args)
        self.replay=collections.deque()
        self.epsilon=0.05
        self.frameNum=10000
        self.batchSize=32
        self.startCount=1000
        self.count=0
        self.dqn=None
        self.synchroStep=100

    def registerInitialState(self,state):
        super().registerInitialState(state)
        walls=state.getWalls()
        width=walls.width
        height=walls.height
        channels=6  # pacman,ghost,wall,food,capsule,scaredghost
        ydim=5 # north,south,west,east,stop
        if self.dqn is None:
            #self.dqn = DeepQNet(width, height, channels,ydim,logdir="../logs/",savedir="../save/")
            self.dqn = DeepQNet(width, height, channels, ydim,logdir="../logs/")

    def getPolicy(self,state,legalActions):
        state=translateState(state)
        state=state[np.newaxis]
        qvalue=self.dqn.qvalue_distribution(state)
        actArray=[]
        for i,v in enumerate(qvalue):
            actArray.append((i,v))
        actArray.sort(key=lambda e:e[1])
        for (idx,act) in enumerate(reversed(actArray)):
            i=act[0]
            if getDirection(i) in legalActions:
                break
        idx=len(actArray)-idx-1
        i=actArray[idx][0]
        action=getDirection(i)
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
        frame=(state,action,nextState,reward)
        self.replay.append(frame)
        if len(self.replay)>self.frameNum:
            self.replay.popleft()
        self.count+=1
        if self.count<self.startCount:
            return
        self.train()

    def train(self):
        batch=random.sample(self.replay,self.batchSize)
        state=[]
        action=[]
        nextState=[]
        reward=[]
        for frame in batch:
            state.append(frame[0])
            action.append(frame[1])
            nextState.append(frame[2])
            reward.append(frame[3])
        qvalue=self.dqn.max_qvalue(nextState)
        self.dqn.train(state,action,reward,qvalue)
