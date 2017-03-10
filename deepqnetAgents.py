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

    def __init__(self,width,height,channels,ydim,logdir=None,savedir=None):
        graph=tf.Graph()
        with graph.as_default():
            self.ydim=ydim
            self.x=tf.placeholder(tf.float32,[None,width,height,channels])
            self.qvalue=tf.placeholder(tf.float32,[None])
            self.reward=tf.placeholder(tf.float32,[None])
            self.actions=tf.placeholder(tf.float32,[None,self.ydim])
            self.keep_prob=tf.placeholder(tf.float32)

            size=4;filters=64
            weight=tf.Variable(tf.random_normal([size,size,channels,filters],stddev=0.01))
            bias=tf.Variable(tf.constant(0.1,shape=[filters]))
            conv=tf.nn.conv2d(self.x,weight,strides=[1,1,1,1],padding="VALID")+bias
            layer=tf.nn.relu(conv)
            #layer=tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            tf.summary.histogram("weight1",weight)
            tf.summary.histogram("bias1",bias)

            size=3;channels=filters;filters=64
            weight=tf.Variable(tf.random_normal([size,size,channels,filters],stddev=0.01))
            bias=tf.Variable(tf.constant(0.1,shape=[filters]))
            conv=tf.nn.conv2d(layer,weight,strides=[1,1,1,1],padding="VALID")+bias
            layer=tf.nn.relu(conv)
            #layer=tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            tf.summary.histogram("weight2",weight)
            tf.summary.histogram("bias2",bias)

            layerShape=layer.get_shape().as_list()
            indim=layerShape[1]*layerShape[2]*layerShape[3];outdim=256
            flat=tf.reshape(layer,[-1,indim])
            weight=tf.Variable(tf.random_normal([indim,outdim]))
            bias=tf.Variable(tf.constant(0.1,shape=[outdim]))
            layer=tf.nn.relu(tf.matmul(flat,weight)+bias)
            layer=tf.nn.dropout(layer,keep_prob=self.keep_prob)
            tf.summary.histogram("weight3",weight)
            tf.summary.histogram("bias3",bias)

            indim=outdim;outdim=self.ydim;
            weight=tf.Variable(tf.random_normal([indim,outdim],stddev=0.01))
            bias=tf.Variable(tf.constant(0.1,shape=[outdim]))
            self.y=tf.matmul(layer,weight)+bias
            tf.summary.histogram("weight4",weight)
            tf.summary.histogram("bias4",bias)

            discount=tf.constant(0.9)
            newQ=self.reward+discount*self.qvalue
            predictQ=tf.reduce_sum(self.y*self.actions,reduction_indices=1)
            loss=tf.reduce_sum(tf.square(newQ-predictQ))
            tf.summary.scalar("loss",loss)
            tf.summary.histogram("Qvalue",predictQ)

            self.global_step=tf.Variable(0,trainable=False)
            optimizer=tf.train.RMSPropOptimizer(0.001,decay=0.99)
            self.rmsprop=optimizer.minimize(loss,global_step=self.global_step)

            self.sess=tf.Session()
            self.sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()
            self.logdir=logdir
            if self.logdir is not None:
                self.writer=tf.summary.FileWriter(self.logdir,self.sess.graph)
                self.summary=tf.summary.merge_all()
            self.savedir=savedir
            if self.savedir is not None:
                savefile=tf.train.latest_checkpoint(self.savedir)
                if savefile is not None:
                    self.saver.restore(self.sess,savefile)

    def train(self,state,actions,reward,qvalue):
        feed_dict={self.x:state,self.qvalue:qvalue,self.actions:actions,self.reward:reward,self.keep_prob:0.5}
        _,step=self.sess.run([self.rmsprop,self.global_step],feed_dict=feed_dict)
        #step=tf.train.global_step(self.sess, self.global_step)
        if self.logdir is not None:
            if step%100==0:
                summary=self.sess.run(self.summary,feed_dict=feed_dict)
                self.writer.add_summary(summary,step)
        if self.savedir is not None:
            if step%1000==0:
                self.saver.save(self.sess, self.savedir, global_step=step)
        return step

    def qvalue_vector(self,batchSize,state):
        feed_dict={self.x:state,self.qvalue:np.zeros(batchSize),self.actions:np.zeros([batchSize,self.ydim]),self.reward:np.zeros(batchSize),self.keep_prob:1.0}
        qvalue=self.sess.run(self.y,feed_dict=feed_dict)
        qvalue=np.amax(qvalue,axis=1)
        return qvalue

    def qvalue_distribution(self,state):
        feed_dict={self.x:state,self.qvalue:np.zeros(1),self.actions:np.zeros([1,self.ydim]),self.reward:np.zeros(1),self.keep_prob:1.0}
        qvalue=self.sess.run(self.y,feed_dict=feed_dict)
        return qvalue[0]

    def copyto(self,other,dir):
        saved=self.saver.save(self.sess,dir)
        other.saver.restore(other.sess,saved)

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
        self.target=None
        self.synchroStep=1000

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
        if self.target is None:
            self.target=DeepQNet(width,height,channels,ydim)

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
        qvalue=self.target.qvalue_vector(self.batchSize,nextState)
        step=self.dqn.train(state,action,reward,qvalue)
        if step%self.synchroStep==0:
            self.dqn.copyto(self.target,"../save/tmp")
