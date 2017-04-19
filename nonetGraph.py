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
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from datetime import datetime
import os

class BonePosition:
    def __del__(self):
        self.sess.close()

    def calclocation(self):#->[vertexNum,matx]
        pose = tf.gather(self.pose, self.index,name="gather_pose")  # [boneNum,matx,maty],[vertexNum,vbnum]->[vertexNum,vbnum,matx,maty]
        bone = tf.gather(self.boneVar, self.index,name="gather_bone") # [boneNum,matx,maty],[vertexNum,vbnum]->[vertexNum,vbnum,matx,maty]
        cat=tf.constant(1.0,dtype=tf.float32,shape=[self.vertexNum,1])
        vertex=tf.concat([self.vertex,cat],axis=1,name="concat_vertex") #[vertexNum,matx+1],makesure maty=matx+1
        cat=tf.reshape(cat,[self.vertexNum,1,1])
        cat=tf.tile(cat,[1,self.vbnum,1],name="tile_cat") #[vertexNum,vbnum,1]

        vertex=tf.reshape(vertex,[self.vertexNum,self.maty,1])
        pose=tf.reshape(pose,[self.vertexNum,self.vbnum*self.matx,self.maty])
        location=tf.matmul(pose,vertex,name="matmul_pose") #[vertexNum,vbnum*matx,1]
        location=tf.reshape(location,[self.vertexNum,self.vbnum,self.matx])

        location=tf.concat([location,cat],axis=2,name="concat_location") #[vertexNum,vbnum,matx+1]
        location=tf.reshape(location,[self.vertexNum,self.vbnum,self.maty,1])
        location=tf.matmul(bone,location,name="matmul_bone") #[vertexNum,vbnum,matx,1]
        location=tf.reshape(location,[self.vertexNum,self.vbnum,self.matx])

        weight=tf.reshape(self.weight,[self.vertexNum,self.vbnum,1])
        location=tf.multiply(location,weight,name="multiply_weight") #[vertexNum,vbnum,matx]
        location=tf.reduce_sum(location,axis=[1],name="sum_weight")#[vertexNum,matx]
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

        self.loc=self.calclocation()#[vertexNum,matx]
        self.loss=self.calcloss(self.loc)
        tf.summary.scalar("loss", self.loss)
        self.optlist=self.optimizers(logdir)

        self.sess=tf.Session()
        self.logdir=logdir
        if self.logdir is not None:
            for i,v in enumerate(self.optlist):
                subdir=self.sublogdir(self.logdir,i)
                v.append(tf.summary.FileWriter(subdir,self.sess.graph,flush_secs=10))
            self.summary=tf.summary.merge_all()
        self.profile=profile

    def optimizers(self,logdir):
        optlist=[]
        trainNum=50
        optlist.append([tf.train.AdamOptimizer(0.001).minimize(self.loss),trainNum])
        if logdir is not None:
            optlist.append([tf.train.GradientDescentOptimizer(0.001).minimize(self.loss), trainNum])
            optlist.append([tf.train.MomentumOptimizer(0.0002,0.9).minimize(self.loss),trainNum])
            optlist.append([tf.train.AdagradOptimizer(0.002).minimize(self.loss),trainNum])
            optlist.append([tf.train.ProximalGradientDescentOptimizer(0.002).minimize(self.loss), trainNum])
            optlist.append([tf.train.ProximalAdagradOptimizer(0.002).minimize(self.loss), trainNum])
            #optlist.append([tf.train.RMSPropOptimizer(0.0002,0.9).minimize(self.loss), trainNum]) #slow
            #optlist.append([tf.train.AdagradDAOptimizer(0.001).minimize(self.loss), trainNum]) #need global_step
            #optlist.append([tf.train.FtrlOptimizer(0.5).minimize(self.loss),trainNum]) #fast but big loss
            #optlist.append([tf.train.AdadeltaOptimizer(0.5).minimize(self.loss),trainNum]) #slow
        return optlist

    def sublogdir(self,logdir,idx):
        now = datetime.now()
        subdir = logdir + "/ev_" + now.strftime("%Y%m%d-%H%M%S")+"_"+str(idx)
        return subdir

    def trainopt(self,feed_dict,optimizer,trainNum,writer):
        options=None
        metadata=None
        if self.profile is not None:
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            metadata=tf.RunMetadata()
            trainNum=2
        self.sess.run([tf.global_variables_initializer(),self.boneInit],feed_dict={self.bone:feed_dict[self.bone]},options=options,run_metadata=metadata)
        for i in range(0,trainNum):
            self.sess.run(optimizer,feed_dict=feed_dict,options=options,run_metadata=metadata)
            self.writeprofile(metadata,i)
            if self.logdir is not None:
                summary=self.sess.run(self.summary,feed_dict=feed_dict,options=options,run_metadata=metadata)
                writer.add_summary(summary,i+1)
        bone=self.sess.run(self.boneVar,options=options,run_metadata=metadata)
        return bone

    def train(self,feed_dict):
        bone=None
        if self.logdir is not None:
            for [opt,num,writer] in self.optlist:
                bone=self.trainopt(feed_dict,opt,num,writer)
        else:
            bone=self.trainopt(feed_dict,self.optlist[0][0],self.optlist[0][1],None)
        return bone

    def writeprofile(self,metadata,idx):
        if self.profile is None:
            return
        tl=timeline.Timeline(metadata.step_stats)
        #ctf=tl.generate_chrome_trace_format(show_dataflow=True,show_memory=True)
        ctf = tl.generate_chrome_trace_format()
        with open(self.profile+str(idx)+".json","w") as f:
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

def DeleteLogDir(logdir):
    import stat
    for f in os.listdir(logdir):
        subdir=os.path.join(logdir,f)
        for f in os.listdir(subdir):
            path=os.path.join(subdir,f)
            if not os.access(path,os.W_OK):
                try:
                    os.chmod(path,stat.S_IWUSR)
                    os.remove(path)
                except Exception as e:
                    print(e)
            else:
                os.remove(path)
        try:
            os.rmdir(subdir)
        except Exception as e:
            print(e)

#bp=BonePosition(featureNum,boneNum,vertexNum)
bp=BonePosition(featureNum,boneNum,vertexNum,"../logs")
#bp=BonePosition(featureNum,boneNum,vertexNum,None,"../timeline")

feed_dict={bp.feature:feature,
           bp.bone:bone,
           bp.pose:pose,
           bp.vertex:vertex,
           bp.weight:weight,
           bp.index:index}

print("begin training")
bone=bp.train(feed_dict)
WriteNewBone(bone)

