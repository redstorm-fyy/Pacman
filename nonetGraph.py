import tensorflow as tf

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

    def __init__(self,featureNum,boneNum,vertexNum):
        self.matx=3
        self.maty=self.matx+1
        self.vbnum=5 #one vertex connects vbnum bones
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

        location=self.calclocation()#[vertexNum,matx]
        loss=self.calcloss(location)
        self.train=tf.train.RMSPropOptimizer(0.1).minimize(loss)

        self.sess=tf.Session()
        init=tf.global_variables_initializer()
        self.sess.run(init)

    def train(self,feed_dict):
        self.sess.run(self.boneInit,feed_dict={self.bone:feed_dict[self.bone]})
        for i in range(0,50):
            self.sess.run(self.train,feed_dict=feed_dict)
        bone=self.sess.run(self.boneVar)
        return bone

bp=BonePosition(11,12,13)

bp.train({bp.feature:{},bp.bone:{},bp.pose:{},bp.vertex:{},bp.boneweight:{},bp.boneindex:{}})