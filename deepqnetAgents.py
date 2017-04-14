import tensorflow as tf
import numpy as np
import collections
import random
import game
import util
from learningAgents import ReinforcementAgent

class QFunction:

    def addConv2d(self,layer, size, channels, filters):
        weight = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.1))
        bias = tf.Variable(tf.constant(0.0, shape=[filters]))
        conv = tf.nn.conv2d(layer, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
        layer = tf.nn.relu(conv)
        self.var.append(weight)
        self.var.append(bias)
        tf.summary.histogram("weight", weight)
        tf.summary.histogram("bias", bias)
        return layer

    def addPooling(self,layer,size):
        return tf.nn.max_pool(layer,ksize=[1,size,size,1],strides=[1,size,size,1],padding="SAME")

    def addLayer(self,layer, indim, outdim, actfn=None):
        weight = tf.Variable(tf.random_normal([indim, outdim],stddev=0.1))
        bias = tf.Variable(tf.constant(0.0, shape=[outdim]))
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

        with tf.name_scope("layers"):
            layer = self.addConv2d(x, 3, channels, 128)
            layer = self.addConv2d(layer, 3, 128, 128)

            layerShape = layer.get_shape().as_list()
            indim = layerShape[1] * layerShape[2] * layerShape[3];
            layer = tf.reshape(layer, [-1, indim])
            layer,self.regular = self.addLayer(layer, indim, 1024, actfn=tf.nn.relu)
            layer = tf.nn.dropout(layer, keep_prob=keep_prob)

            self.y,_ = self.addLayer(layer, 1024, ydim)

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
        self.terminal=tf.placeholder(tf.float32,[None])

        self.qfunc=QFunction(self.x,channels,self.ydim,self.keep_prob)
        self.tfunc=QFunction(self.x,channels,self.ydim,self.keep_prob)

        with tf.name_scope("synchro"):
            self.sychro=[]
            for i,var in enumerate(self.qfunc.var):
                self.sychro.append(tf.assign(self.tfunc.var[i],var))

        with tf.name_scope("loss"):
            discount=0.9*(1.0-self.terminal)
            newQ=self.reward+discount*self.qvalue
            predictQ=tf.reduce_sum(self.qfunc.y*self.actions,reduction_indices=1)
            loss=tf.reduce_sum(tf.square(newQ-predictQ)+0.001*tf.nn.l2_loss(self.qfunc.regular))
            tf.summary.scalar("loss",loss)
            tf.summary.scalar("qvalue",tf.reduce_mean(predictQ,reduction_indices=0))
            terminalmean=tf.reduce_mean(self.terminal, reduction_indices=0)
            tf.summary.scalar("terminal",terminalmean)

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
                self.restored=True
            else:
                self.restored=False

    def train(self,state,actions,reward,qvalue,terminal):
        feed_dict={self.x:state,self.actions:actions,self.reward:reward,self.qvalue:qvalue,self.terminal:terminal,self.keep_prob:0.5}
        _,step=self.sess.run([self.rmsprop,self.global_step],feed_dict=feed_dict)
        #step=tf.train.global_step(self.sess, self.global_step)
        if step%1000==0:
            self.sess.run(self.sychro)
        if self.logdir is not None:
            if step%1000==0:
                summary=self.sess.run(self.summary,feed_dict=feed_dict)
                self.writer.add_summary(summary,step)
        if self.savedir is not None:
            if step%1000==0:
                self.saver.save(self.sess, self.savedir, global_step=step)
        return step

    def max_qvalue(self,state):
        feed_dict={self.x:state,self.keep_prob:1.0}
        qvalue=self.sess.run(self.tfunc.y,feed_dict=feed_dict) # target network
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

Channels=6  # pacman,ghost,wall,food,capsule,scaredghost

def translateState(state):
    walls=state.getWalls()
    width=walls.width
    height=walls.height
    tensor=np.zeros([width,height,Channels],dtype=int)
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
        self.terminal=True
        super().final(state)
        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('\treplay',len(self.replay),'success',self.success)


    def __init__(self,**args):
        super().__init__(**args)
        self.replay=collections.deque()
        self.epsilon=0.1
        self.frameNum=500000
        self.batchSize=32
        self.startCount=10000
        self.count=0
        self.dqn=None
        self.terminal=False
        self.success=0

    def registerInitialState(self,state):
        super().registerInitialState(state)
        if self.numTraining==0:
            self.epsilon=0
        self.terminal=False
        walls=state.getWalls()
        width=walls.width
        height=walls.height
        ydim=5 # north,south,west,east,stop
        if self.dqn is None:
            self.dqn = DeepQNet(width, height, Channels,ydim,logdir="../logs/",savedir="../save/")
            #self.dqn = DeepQNet(width, height, Channels, ydim,logdir="../logs/")
            if self.dqn.restored:
                self.startCount=self.frameNum

    def getPolicy(self,state,legalActions):
        state=translateState(state)
        #state=np.concatenate((state,state),axis=2)
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
        if self.terminal and reward>0:
            self.success=self.success+1
        if reward==500:
            reward=100 # win
        elif reward==200:
            reward=50 # eaten ghost
        elif reward==-500:
            reward=-100 # lose
        state=translateState(state)
        nextState=translateState(nextState)
        action=translateAction(action)
        terminal=False
        if self.terminal and reward<0:
            terminal=True
        frame=(state,action,nextState,reward,terminal)
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
        terminal=[]
        for frame in batch:
            state.append(frame[0])
            #state.append(np.concatenate((frame[0], frame[2]), axis=2))#concat state and nextState to be real state
            action.append(frame[1])
            nextState.append(frame[2])
            #nextState.append(np.concatenate((frame[2], frame[2]), axis=2))
            reward.append(frame[3])
            terminal.append(frame[4])
        qvalue=self.dqn.max_qvalue(nextState)
        self.dqn.train(state,action,reward,qvalue,terminal)
