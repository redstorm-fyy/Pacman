'''
this is the calculation of bones
Vector3 Mul(Matrix4x4 matrix, Vector3 point)
{
    Vector4 p = point;
    p.w = 1;
    Vector3 r;
    r.x = Vector4.Dot(p, new Vector4(matrix.m00, matrix.m01, matrix.m02, matrix.m03));
    r.y = Vector4.Dot(p, new Vector4(matrix.m10, matrix.m11, matrix.m12, matrix.m13));
    r.z = Vector4.Dot(p, new Vector4(matrix.m20, matrix.m21, matrix.m22, matrix.m23));
    return r;
}
Vector3 CalculateLocation(Mesh mesh,int vertexIndex)
{
    Vector3 point = mesh.vertices[vertexIndex];
    var bw = mesh.boneWeights[vertexIndex];
    Vector3 result = Vector3.zero;
    result += Mul(bones[bw.boneIndex0].localToWorldMatrix, Mul(mesh.bindposes[bw.boneIndex0], point)) * bw.weight0;
    result += Mul(bones[bw.boneIndex1].localToWorldMatrix, Mul(mesh.bindposes[bw.boneIndex1], point)) * bw.weight1;
    result += Mul(bones[bw.boneIndex2].localToWorldMatrix, Mul(mesh.bindposes[bw.boneIndex2], point)) * bw.weight2;
    result += Mul(bones[bw.boneIndex3].localToWorldMatrix, Mul(mesh.bindposes[bw.boneIndex3], point)) * bw.weight3;
    return result;
}

optimizer test
optlist.append([tf.train.GradientDescentOptimizer(0.003).minimize(self.loss), trainNum])
optlist.append([tf.train.AdamOptimizer(0.001).minimize(self.loss), trainNum]) # precise but slower than GradientDescentOptimizer
optlist.append([tf.train.AdagradOptimizer(0.002).minimize(self.loss),trainNum])# sometimes big loss
optlist.append([tf.train.MomentumOptimizer(0.0002,0.9).minimize(self.loss),trainNum]) # sometimes big loss
optlist.append([tf.train.ProximalGradientDescentOptimizer(0.003).minimize(self.loss), trainNum]) # nearly same as GradientDescentOptimizer
optlist.append([tf.train.ProximalAdagradOptimizer(0.002).minimize(self.loss), trainNum]) # nearly same as AdagradOptimizer
optlist.append([tf.train.RMSPropOptimizer(0.0002,0.9).minimize(self.loss), trainNum]) #slow
optlist.append([tf.train.AdadeltaOptimizer(0.5).minimize(self.loss),trainNum]) #slow
optlist.append([tf.train.AdagradDAOptimizer(0.001).minimize(self.loss), trainNum]) #need global_step
optlist.append([tf.train.FtrlOptimizer(0.5).minimize(self.loss),trainNum]) #fast but big loss

'''

import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from datetime import datetime

class BonePosition:
    def __del__(self):
        self.sess.close()

    def calclocation(self):#->[vertexNum,matx]
        pose = tf.gather(self.pose, self.index,name="gather_pose")  # [boneNum,matx,maty],[vertexNum,vbnum]->[vertexNum,vbnum,matx,maty]
        bone = tf.gather(self.boneVar, self.index,name="gather_bone") # [boneNum,matx,maty],[vertexNum,vbnum]->[vertexNum,vbnum,matx,maty]
        cat=tf.constant(1.0,dtype=tf.float32,shape=[self.vertexNum,1])
        vertex=tf.concat([self.vertex,cat],axis=1,name="concat_vertex") #[vertexNum,matx+1],makesure maty=matx+1

        vertex=tf.reshape(vertex,[self.vertexNum,self.maty,1])
        pose=tf.reshape(pose,[self.vertexNum,self.vbnum*self.matx,self.maty])
        location=tf.matmul(pose,vertex,name="matmul_pose") #[vertexNum,vbnum*matx,1]
        location=tf.reshape(location,[self.vertexNum,self.vbnum,self.matx])

        cat=tf.reshape(cat,[self.vertexNum,1,1])
        cat=tf.tile(cat,[1,self.vbnum,1],name="tile_cat") #[vertexNum,vbnum,1]
        location=tf.concat([location,cat],axis=2,name="concat_location") #[vertexNum,vbnum,matx+1]
        location=tf.reshape(location,[self.vertexNum,self.vbnum,self.maty,1])
        location=tf.matmul(bone,location,name="matmul_bone") #[vertexNum,vbnum,matx,1]
        location=tf.reshape(location,[self.vertexNum,self.vbnum,self.matx])

        weight=tf.reshape(self.weight,[self.vertexNum,self.vbnum,1])
        location=tf.multiply(location,weight,name="multiply_weight") #[vertexNum,vbnum,matx]
        location=tf.reduce_sum(location,axis=1,name="sum_weight")#[vertexNum,matx]
        return location

    def calcloss(self,location): #[vertexNum,matx],[featureNum,matx]->[]
        ab=tf.matmul(location,tf.transpose(self.feature),name="matmul_loss") #[vertexNum,featureNum]
        a=tf.reduce_sum(location*location,axis=1,name="sum_location") #[vertexNum]
        b=tf.reduce_sum(self.feature*self.feature,axis=1,name="sum_feature") #[featureNum]
        a=tf.reshape(a,[self.vertexNum,1])
        b=tf.reshape(b,[1,self.featureNum])
        loss=a-2*ab+b #[vertexNum,featureNum],Distance(A,B)=(A-B)(A-B)=AA-2AB+BB
        loss=tf.reduce_sum(tf.reduce_min(loss,axis=0,name="min_distance"),name="sum_loss")
        return loss

    def body(self,i,optimizer):
        loss=self.calcloss(self.calclocation())
        opt=optimizer.minimize(loss)
        return tf.tuple([i + 1], control_inputs=[opt])

    def __init__(self,featureNum,boneNum,vertexNum,logdir=None,profile=None):
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

        self.trainNum=20
        optimizer=tf.train.GradientDescentOptimizer(0.003)
        #optimizer=tf.train.RMSPropOptimizer(0.0002,0.9)
        if logdir is not None:
            loss = self.calcloss(self.calclocation())
            self.summary=tf.summary.scalar("loss",loss)
            self.y = optimizer.minimize(loss)
        else:
            self.y=tf.while_loop(lambda i:i<self.trainNum,lambda i:self.body(i,optimizer),[tf.constant(0)],parallel_iterations=10)

        self.sess=tf.Session()
        self.logdir=logdir
        if self.logdir is not None:
            subdir = self.sublogdir(self.logdir)
            self.writer=tf.summary.FileWriter(subdir, self.sess.graph,flush_secs=1)
        self.profile=profile

    def train(self,feed_dict):
        options=None
        metadata=None
        if self.profile is not None:
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            metadata=tf.RunMetadata()
        self.sess.run([tf.global_variables_initializer(),self.boneInit],feed_dict={self.bone:feed_dict[self.bone]},options=options,run_metadata=metadata)
        if self.logdir is not None:
            summary = self.sess.run(self.summary, feed_dict=feed_dict,options=options, run_metadata=metadata)
            self.writer.add_summary(summary, 0)
            for i in range(0,self.trainNum):
                self.sess.run(self.y,feed_dict=feed_dict,options=options,run_metadata=metadata)
                self.writeprofile(metadata)
                summary=self.sess.run(self.summary,feed_dict=feed_dict,options=options,run_metadata=metadata)
                self.writer.add_summary(summary,i+1)
        else:
            self.sess.run(self.y,feed_dict=feed_dict,options=options,run_metadata=metadata)
            self.writeprofile(metadata)
        bone=self.sess.run(self.boneVar,options=options,run_metadata=metadata)
        return bone

    def sublogdir(self,logdir):
        now = datetime.now()
        subdir = logdir + "/ev_" + now.strftime("%Y%m%d-%H%M%S")
        return subdir

    def writeprofile(self,metadata):
        if self.profile is None:
            return
        tl=timeline.Timeline(metadata.step_stats)
        #ctf=tl.generate_chrome_trace_format(show_dataflow=True,show_memory=True)
        ctf = tl.generate_chrome_trace_format()
        with open(self.profile+".json","w") as f:
            f.write(ctf)

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

graphdir="../graph"

def ReadBone():
    with open(graphdir+"/bones.txt","rb") as f:
        return ReadMatrixlist(f)
def ReadPose():
    with open(graphdir+"/bindpose.txt","rb") as f:
        return ReadMatrixlist(f)

def ReadVertex():
    with open(graphdir+"/vertices.txt","rb") as f:
        return ReadVectorlist(f)
def ReadFeature():
    with open(graphdir+"/new_vertices.txt","rb") as f:
        return ReadVectorlist(f)

def ReadIndexAndWeight():
    with open(graphdir+"/weights.txt","rb") as f:
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

def WriteNewBone(bone):
    with open(graphdir+"/new_bone.txt","wb") as f:
        for b in bone:
            b=np.array(b)
            b.tofile(f,sep="\t")
            f.write(b"\r\n")

tm1=datetime.now()
print("version",tf.__version__,tm1)

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

tm2=datetime.now()
print("start",tm2,tm2-tm1,[featureNum,boneNum,poseNum,vertexNum,indexNum,weightNum])

#bp=BonePosition(featureNum,boneNum,vertexNum)
bp=BonePosition(featureNum,boneNum,vertexNum,"../logs")
#bp=BonePosition(featureNum,boneNum,vertexNum,None,"../timeline")

feed_dict={bp.feature:feature,
           bp.bone:bone,
           bp.pose:pose,
           bp.vertex:vertex,
           bp.weight:weight,
           bp.index:index}

tm1=tm2;tm2=datetime.now()
print("train",tm2,tm2-tm1)

bone=bp.train(feed_dict)
tm1=tm2;tm2=datetime.now()
print("end",tm2,tm2-tm1)
WriteNewBone(bone)

#print(bp.sess.run(bp.loss,feed_dict=feed_dict))