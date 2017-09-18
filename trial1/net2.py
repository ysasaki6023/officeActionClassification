# -*- coding: utf-8 -*-
import os,sys,argparse,re,string,logging,random,csv,glob,cv2
#import tensorflow as tf
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities
from gensim.models import word2vec
from keras.layers import Input, Dense, LSTM, merge, Lambda, GRU, Dot, Reshape, Concatenate, Flatten, Dropout, Bidirectional, TimeDistributed, Activation
from keras.utils import to_categorical
from keras.constraints import min_max_norm
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.applications.inception_v3 import InceptionV3, preprocess_input

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

class net(object):
    def __init__(self,isTraining=True,nBatch=1,nGRU=4096,nLength=60,learnRate=1e-5,saveFolder="save"):
        #self.actDict = {1:"drinking",2:"eating",3:"reading",4:"calling",5:"writing",6:"typing",7:"cleaning",8:"cheering",9:"sitting",10:"throwing",11:"gaming",12:"sleeping",13:"walking",14:"playing music",15:"standing up", 16:"sitting down"}
        self.actDict = {1:"drinking",2:"eating",3:"reading",4:"calling",5:"writing",6:"typing",13:"walking"}
        self.convTable = self.actDict.keys()
        self.nColor = 3
        self.sizeX = 192 # 640
        self.sizeY = 144 # 480
        self.fps = 30
        self.fileList = glob.glob("../data/*.avi")
        self.nActivities = len(self.convTable)
        self.doFineTune = args.doFineTune

        self.cut_before = 1.0 # sec
        self.cut_after  = 1.0 # sec

        self.nBatch  = nBatch
        self.nGRU    = nGRU
        self.nLength = nLength
        self.learnRate = learnRate
        self.saveFolder = saveFolder
        self.isTraining = isTraining
        #if isTraining:
        #self.buildModel()
        return

    def cvtPath2Cls(self,path):
        ttt = int(re.search(r"a(\d\d)_",os.path.basename(path)).group(1))
        if not ttt in self.convTable:
            return None
        else:
            return self.convTable.index(ttt)

    def cvtCls2act(self,cls):
        return self.actDict[self.convTable[cls]]

    def yieldOne(self,mode="all"):
        addNoise = False
        fileList = self.fileList
        while True:
            batchX = np.zeros( (self.nBatch, self.nLength, self.sizeY, self.sizeX, self.nColor), dtype=np.float32)
            batchT = np.zeros( (self.nBatch, self.nActivities)        , dtype=np.int32)
            bIdx = 0
            while bIdx<self.nBatch:
                path = random.choice(fileList)
                #if not self.cvtPath2Cls(path) in self.actDict.keys(): continue
                clsIdx = self.cvtPath2Cls(path)
                if not clsIdx: continue
                batchT[bIdx] = to_categorical(clsIdx,num_classes=self.nActivities)
                mov = cv2.VideoCapture(path)
                fps         = float(mov.get(cv2.CAP_PROP_FPS))
                totalLength = float(mov.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
                shift_time = random.uniform(self.cut_before,totalLength - self.cut_after - float(self.nLength)/fps - 0.1 )
                shift_w, shift_h = 0.1, 0.1
                img_w, img_h = mov.get(cv2.CAP_PROP_FRAME_WIDTH),mov.get(cv2.CAP_PROP_FRAME_HEIGHT)
                shift_x = random.uniform(-self.sizeX*shift_w,+self.sizeX*shift_w)
                shift_y = random.uniform(-self.sizeY*shift_h,+self.sizeY*shift_h)
                mov.set(cv2.CAP_PROP_POS_MSEC,shift_time*1000)
                for tIdx in range(self.nLength):
                    ret, frame = mov.read()
                    if addNoise: frame += np.random.normal(0,1,frame.shape).astype(np.uint8)
                    #cv2.imshow("frame",frame)
                    #cv2.waitKey(int(1))
                    if not ret: continue # 基本的に、長さ上は絶対に終端までは来ないので、retが帰ってこないのは単なるエラー
                    frame = cv2.resize(frame,(int(self.sizeX*(1.+2*shift_w)),int(self.sizeY*(1.+2*shift_h))),interpolation=cv2.INTER_CUBIC)
                    sizeH, sizeW, _ = frame.shape
                    posX1 = int(sizeW/2.-self.sizeX/2.) + int(shift_x)
                    posY1 = int(sizeH/2.-self.sizeY/2.) + int(shift_y)
                    posX2 = posX1 + self.sizeX
                    posY2 = posY1 + self.sizeY
                    frame = frame[posY1:posY2,posX1:posX2]
                    batchX[bIdx,tIdx] = frame
                mov.release()
                bIdx += 1
            yield ({"inX":preprocess_input(batchX)},{"cls":batchT})

    def reloadModel(self,fPath):
        self.model.load_weights(fPath)
        print "model loaded from %s"%fPath
        return

    def loadModel(self,fPath):
        self.model = load_model(fPath)
        print "model loaded from %s"%fPath
     
    def buildModel(self):
        #K.set_learning_phase(True)

        input_shape = (self.nLength,self.sizeY,self.sizeX,self.nColor)
        inX = Input(input_shape,name="inX")
        cnnModel = InceptionV3(weights="imagenet", include_top=False, pooling="max")

        visModelAll = TimeDistributed(cnnModel)(inX)
        visFlatten  = Reshape((self.nLength,-1))(visModelAll)
        gruModel    = GRU(self.nGRU)(visFlatten)
        gruModel    = Dropout(0.5)(gruModel)
        logits      = Dense(self.nActivities)(gruModel)
        output      = Activation("softmax",name="cls")(logits)

        model = Model(inputs=inX, outputs=output)

        model.compile(loss="categorical_crossentropy",optimizer=Adam(self.learnRate),metrics=["accuracy"])

        if not self.doFineTune:
            for layer in cnnModel.layers:
                layer.trainable = False

        model.summary()
        self.model = model

        return

    def train(self):
        cp_cb = ModelCheckpoint(filepath = self.saveFolder+"/weights.{epoch:02d}.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto')
        tb_cb = TensorBoard(log_dir=self.saveFolder, histogram_freq=1)
        lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-10, epsilon=0.01, verbose=1)

        self.model.fit_generator(generator=self.yieldOne("all"),
                                epochs=100000000,
                                callbacks=[cp_cb,tb_cb,lr_cb],
                                steps_per_epoch=len(self.fileList),
                                use_multiprocessing=True, 
                                max_queue_size=10, 
                                workers=1)

    def test(self,movPath,outPath=None):
        actDict = self.actDict
        if movPath=="0":
            movFileList = [0]
        else:
            movFileList = glob.glob(movPath)

        if outPath:
            fourcc = cv2.VideoWriter_fourcc(*"H264")
            out = cv2.VideoWriter(outPath,fourcc, 30.0, (self.sizeX*3,self.sizeY*3))
        for movFile in movFileList:
            mov = cv2.VideoCapture(movFile)
            inX = np.zeros((1,self.nLength,self.sizeY,self.sizeX,3))
            if movFile==0:
                fps = 30
                totalLength = 1000000000
            else:
                fps         = float(mov.get(cv2.CAP_PROP_FPS))
                totalLength = int(mov.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in xrange(totalLength):
                ret, frame = mov.read()
                if not ret:
                    print "ret is None"
                    continue
                if movFile==0:
                    _,w,_ = frame.shape
                    aspect = 4./3.
                    h = w / aspect
                    frame = frame[:,int(w/2-h/2):int(w/2+h/2)]
                frame = cv2.resize(frame,(self.sizeX,self.sizeY))
                inX[:-1] = inX[1:]
                inX[-1]  = frame
                inX = preprocess_input(inX)
                t = self.model.predict(x=inX,batch_size=1)
                v = t[0]
                t = self.cvtCls2act(np.argmax(t[0]))
                print t,v
                frame = cv2.resize(frame,(self.sizeX*3,self.sizeY*3))
                cv2.putText(frame,t,(10,30),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 2)
                cv2.imshow("frame",frame)
                if outPath: out.write(frame)
                cv2.waitKey(int(1000/fps))
            mov.release()
        if outPath:
            out.release()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nBatch" ,"-b",dest="nBatch",type=int,default=6)
    parser.add_argument("--nGRU"   ,"-g",dest="nGRU"  ,type=int,default=128)
    parser.add_argument("--nLength","-l",dest="nLength"  ,type=int,default=30)
    parser.add_argument("--learnRate",dest="learnRate"  ,type=float,default=1e-5)
    parser.add_argument("--doFineTune","-f",dest="doFineTune"  ,action="store_true")
    parser.add_argument("--reload","-r",dest="reload"  ,type=str,default=None)
    parser.add_argument("--saveFolder","-s",dest="saveFolder"  ,type=str,default="save")
    parser.add_argument("--test","-t",dest="test"  ,type=str,default=None)
    parser.add_argument("--testSave",dest="testSave"  ,type=str,default=None)
    args = parser.parse_args()

    if args.test:
        assert args.reload, "please set model to use"
        n = net(nBatch=1,
                nGRU=args.nGRU,
                nLength=args.nLength,
                learnRate=0,
                isTraining=False,
                saveFolder=None)
        #n.reloadModel(args.reload)
        n.loadModel(args.reload)
        n.test(args.test,args.testSave)
    else:
        n = net(nBatch=args.nBatch,
                nGRU=args.nGRU,
                nLength=args.nLength,
                learnRate=args.learnRate,
                saveFolder=args.saveFolder)
        n.buildModel()
        if args.reload:
            n.reloadModel(args.reload)
        n.train()
