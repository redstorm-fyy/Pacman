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
        pose = tf.gather(self.pose, self.boneindex)  # [boneNum,matx,maty],[vertexNum,vbnum]->[vertexNum,vbnum,matx,maty]
        bone = tf.gather(self.boneVar, self.boneindex) # [boneNum,matx,maty],[vertexNum,vbnum]->[vertexNum,vbnum,matx,maty]
        vertex=tf.reshape(self.vertex,[self.vertexNum,1,self.maty])
        vertex=tf.tile(vertex,[1,self.vbnum,1])#[vertexNum,vbnum,maty]
        cat=tf.slice(vertex,[0,0,self.maty-1],[self.vertexNum,self.vbnum,1])# slice vertex.w [vertexNum,vbnum,1]
        location = self.matmulvector(pose, vertex)  # [vertexNum,vbnum,matx]
        location = tf.concat([location, cat], axis=2)  # [vertexNum,vbnum,matx+1]
        location = self.matmulvector(bone, location)  #  makesure matx+1==maty, [vertexNum,vbnum,matx]
        weight=tf.reshape(self.boneweight,[self.vertexNum,self.vbnum,1])
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
        self.speed=0.1
        self.trainNum=50

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
        self.vertex=tf.placeholder(tf.float32,shape=[self.vertexNum,self.maty])
        self.boneweight=tf.placeholder(tf.float32,shape=[self.vertexNum,self.vbnum])
        self.boneindex=tf.placeholder(tf.int32,shape=[self.vertexNum,self.vbnum])

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

bp=BonePosition(1,1,1,"../logs")
feed_dict={bp.feature:[[330.0,900.0,1600.0]],
           bp.bone:[[[1.0,2.0,3.0,4.0],
                    [5.0, 6.0, 7.0,8.0],
                    [9.0, 10.0, 11.0,12.0]]],
           bp.pose:[[[1.1,2.1,3.1,4.1],
                    [5.1, 6.1, 7.1,8.1],
                    [9.1, 10.1, 11.1,12.1]]],
           bp.vertex:[[1.3,2.3,3.3,1.0]],
           bp.boneweight:[[0.6,0.2,0.1,0.1]],
           bp.boneindex:[[0,0,0,0]]}

print(bp.location(feed_dict))
bone=bp.train(feed_dict)
print(bone)
feed_dict[bp.bone]=bone
print(bp.location(feed_dict))
