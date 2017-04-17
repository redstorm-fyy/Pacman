import tensorflow as tf
import numpy as np

class BonePosition:
    def __del__(self):
        self.sess.close()

    def matmulvector(self,mat,vec):#[vertexNum,vbnum,matx,maty]*[vertexNum,vbnum,maty]->[vertexNum,vbnum,matx]
        vec=tf.reshape(vec,[self.vertexNum,self.vbnum,self.maty,1])
        result=tf.matmul(mat,vec)#[vertexNum,vbnum,matx,1]
        result=tf.reshape(result,[self.vertexNum,self.vbnum,self.matx])
        return result

    def calclocation(self):#->[vertexNum,matx]
        pose = tf.gather(self.pose, self.index)  # [boneNum,matx,maty],[vertexNum,vbnum]->[vertexNum,vbnum,matx,maty]
        bone = tf.gather(self.boneVar, self.index) # [boneNum,matx,maty],[vertexNum,vbnum]->[vertexNum,vbnum,matx,maty]
        cat=tf.constant(1.0,dtype=tf.float32,shape=[self.vertexNum,1])
        vertex=tf.concat([self.vertex,cat],axis=1)#[vertexNum,matx+1],makesure maty=matx+1
        vertex=tf.reshape(vertex,[self.vertexNum,1,self.maty])
        vertex=tf.tile(vertex,[1,self.vbnum,1])#[vertexNum,vbnum,maty]
        cat=tf.slice(vertex,[0,0,self.maty-1],[self.vertexNum,self.vbnum,1])# slice vertex.w [vertexNum,vbnum,1]
        location = self.matmulvector(pose, vertex)  # [vertexNum,vbnum,matx]
        location = tf.concat([location, cat], axis=2)  # [vertexNum,vbnum,matx+1]
        location = self.matmulvector(bone, location)  #  makesure matx+1==maty, [vertexNum,vbnum,matx]
        weight=tf.reshape(self.weight,[self.vertexNum,self.vbnum,1])
        location=location*weight #[vertexNum,vbnum,matx]
        location=tf.reduce_sum(location,axis=[1])#[vertexNum,matx]
        return location

    def calcloss(self,location):
        location=tf.expand_dims(location,axis=0) #[1,vertexNum,matx]
        feature=tf.expand_dims(self.feature,axis=1) #[featureNum,1,matx]
        loss=tf.reduce_sum(tf.squared_difference(location,feature),2) #[featureNum,vertexNum]
        loss=tf.reduce_sum(tf.reduce_min(loss,axis=1))
        return loss

    def __init__(self,featureNum,boneNum,vertexNum,logdir=None):
        self.speed=0.0001
        self.trainNum=200

        self.matx=3
        self.maty=self.matx+1
        self.vbnum=4 #one vertex connects vbnum bones
        self.featureNum=featureNum
        self.boneNum=boneNum
        self.vertexNum=vertexNum

        self.feature=tf.placeholder(tf.float32,shape=[self.featureNum,self.matx])
        self.bone=tf.placeholder(tf.float32,shape=[self.boneNum,self.matx,self.maty])
        self.boneVar=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[self.boneNum,self.matx,self.maty]))
        self.boneInit=tf.assign(self.boneVar,self.bone)

        self.pose=tf.placeholder(tf.float32,shape=[self.boneNum,self.matx,self.maty])
        self.vertex=tf.placeholder(tf.float32,shape=[self.vertexNum,self.matx])
        self.weight=tf.placeholder(tf.float32,shape=[self.vertexNum,self.vbnum])
        self.index=tf.placeholder(tf.int32,shape=[self.vertexNum,self.vbnum])

        self.loc=self.calclocation()#[vertexNum,matx]
        self.loss=self.calcloss(self.loc)
        self.opt=tf.train.RMSPropOptimizer(self.speed).minimize(self.loss)
        tf.summary.scalar("loss", self.loss)

        self.sess=tf.Session()
        init=tf.global_variables_initializer()
        self.sess.run(init)

        self.logdir=logdir
        if self.logdir is not None:
            self.writer=tf.summary.FileWriter(self.logdir,self.sess.graph)
            self.summary=tf.summary.merge_all()


    def train(self,feed_dict):
        self.sess.run(self.boneInit,feed_dict={self.bone:feed_dict[self.bone]})
        for i in range(0,self.trainNum):
            self.sess.run(self.opt,feed_dict=feed_dict)
            if self.logdir is not None:
                summary=self.sess.run(self.summary,feed_dict=feed_dict)
                self.writer.add_summary(summary,i+1)

        bone=self.sess.run(self.boneVar)
        return bone

    def location(self,feed_dict):
        self.sess.run(self.boneInit,feed_dict={self.bone:feed_dict[self.bone]})
        return self.sess.run(self.loc,feed_dict=feed_dict)


def ReadMatrixlist(f):
    mlist=[]
    lines=f.readlines()
    for s in lines:
        numArray=s.split(b"\t")
        mat=np.zeros([3,4],dtype=np.float32)
        for i in range(0,3):
            for j in range(0,4):
                mat[i][j]=numArray[i*4+j]
        mlist.append(mat)
    return mlist

def ReadVectorlist(f):
    vlist = []
    lines = f.readlines()
    for s in lines:
        numArray = s.split(b"\t")
        vec = np.zeros([3], dtype=np.float32)
        for i in range(0, 3):
            vec[i] = numArray[i]
        vlist.append(vec)
    return vlist

def ReadBone():
    with open("../graph/bones.txt","rb") as f:
        return ReadMatrixlist(f)
def ReadPose():
    with open("../graph/bindpose.txt","rb") as f:
        return ReadMatrixlist(f)

def ReadVertex():
    with open("../graph/vertices.txt","rb") as f:
        return ReadVectorlist(f)
def ReadFeature():
    with open("../graph/new_vertices.txt","rb") as f:
        return ReadVectorlist(f)

def ReadIndexAndWeight():
    with open("../graph/weights.txt","rb") as f:
        ilist=[]
        wlist=[]
        lines=f.readlines()
        for s in lines:
            numArray=s.split(b"\t")
            index=np.zeros([4],dtype=np.int32)
            weight=np.zeros([4],dtype=np.float32)
            for i in range(0,4):
                index[i]=numArray[i*2]
                weight[i]=numArray[i*2+1]
            ilist.append(index)
            wlist.append(weight)
        return ilist,wlist

def Writebone(bone):
    with open("../graph/new_bone.txt","wb") as f:
        for b in bone:
            b=np.array(b)
            b.tofile(f,sep="\t")
            f.write(b"\r\n")

bone=ReadBone()
pose=ReadPose()
vertex=ReadVertex()
index,weight=ReadIndexAndWeight()
feature=ReadFeature()

featureNum=len(feature)
boneNum=len(bone)
poseNum=len(pose)
vertexNum=len(vertex)
indexNum=len(index)
weightNum=len(weight)

print(featureNum,boneNum,poseNum,vertexNum,indexNum,weightNum)

bp=BonePosition(featureNum,boneNum,vertexNum,"../logs")
feed_dict={bp.feature:feature,
           bp.bone:bone,
           bp.pose:pose,
           bp.vertex:vertex,
           bp.weight:weight,
           bp.index:index}

print("begin training")
#print(bp.location(feed_dict))
bone=bp.train(feed_dict)
Writebone(bone)
#print(bone)
#feed_dict[bp.bone]=bone
#print(bp.location(feed_dict))
