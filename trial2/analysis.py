# -*- coding: utf-8 -*-
import os,sys,argparse,re,string,logging,random,csv,glob,cv2,util,math
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities
from gensim.models import word2vec
from keras.layers import Input, Dense, LSTM, merge, Lambda, GRU, Dot, Reshape, Concatenate, Flatten, Dropout, Bidirectional, TimeDistributed, Activation, Conv3D, MaxPooling3D, GlobalMaxPooling2D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.constraints import min_max_norm
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from config_reader import config_reader

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

def relu(x): 
    return Activation('relu')(x)

def conv(x, nf, ks, name):
    x1 = Conv2D(nf, (ks, ks), padding='same', name=name, trainable=False)(x)
    return x1

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block(x):
     
    # Block 1
    x = conv(x, 64, 3, "conv1_1")
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1")
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")
    
    # Block 3
    x = conv(x, 256, 3, "conv3_1")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_2")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_3")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_4")
    x = relu(x)    
    x = pooling(x, 2, 2, "pool3_1")
    
    # Block 4
    x = conv(x, 512, 3, "conv4_1")
    x = relu(x)    
    x = conv(x, 512, 3, "conv4_2")
    x = relu(x)    
    
    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM")
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM")
    x = relu(x)
    
    return x

def stage1_block(x, num_p, branch):
    
    # Block 1        
    x = conv(x, 128, 3, "conv5_1_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_2_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_3_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 512, 1, "conv5_4_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, num_p, 1, "conv5_5_CPM_L%d" % branch)
    
    return x

def stageT_block(x, num_p, stage, branch):
        
    # Block 1        
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))
    
    return x

class net(object):
    def __init__(self,isTraining=True,nBatch=1,nLength=60,learnRate=1e-5,saveFolder="save",fileListCSV="traintest.csv"):
        #self.actDict = {1:"drinking",2:"eating",3:"reading",4:"calling",5:"writing",6:"typing",7:"cleaning",8:"cheering",9:"sitting",10:"throwing",11:"gaming",12:"sleeping",13:"walking",14:"playing music",15:"standing up", 16:"sitting down"}
        self.actDict = {1:"drinking",2:"eating",3:"reading",4:"calling",5:"writing",6:"typing",13:"walking"}
        self.convTable = self.actDict.keys()
        self.nColor = 3
        self.sizeX = 192 # 640
        self.sizeY = 144 # 480
        self.fps = 30
        self.nActivities = len(self.convTable)
        self.doFineTune = args.doFineTune
        if not os.path.exists(fileListCSV):
            self.fileList = {}
            self.fileList["all"] = glob.glob("../../data/*.avi")
            self.fileList["train"] = random.sample(self.fileList["all"],k=int(len(self.fileList["all"])*0.9))
            self.fileList["test"]  = [x for x in self.fileList["all"] if not x in self.fileList["train"]]
            with open(fileListCSV,"w") as f:
                for x in self.fileList["train"]:
                    f.write("train,%s\n"%x)
                for x in self.fileList["test"]:
                    f.write("test,%s\n"%x)
        else:
            self.fileList = {"all":[],"train":[],"test":[]}
            with open(fileListCSV) as f:
                for line in f:
                    line = line.strip()
                    line = line.split(",")
                    self.fileList[line[0]].append(line[1])
                    self.fileList["all"].append(line[1])

        self.cut_before = 1.0 # sec
        self.cut_after  = 1.0 # sec

        self.nBatch  = nBatch
        self.nLength = nLength
        self.learnRate = learnRate
        self.saveFolder = saveFolder
        self.isTraining = isTraining

        self.weights_path = "model/model.h5"

        return

    def prepareImg(self,oriImg):
        param, model_params = config_reader()
        multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']]
        imageToTest = oriImg
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])        
        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2))/256 - 0.5; # required shape (1, width, height, channels) 
        return input_img

    def cvtPath2Cls(self,path):
        ttt = int(re.search(r"a(\d\d)_",os.path.basename(path)).group(1))
        if not ttt in self.convTable:
            return None
        else:
            return self.convTable.index(ttt)

    def cvtCls2act(self,cls):
        return self.actDict[self.convTable[cls]]

    def loadOne(self,mode="all"):
        addNoise = False
        fileList = self.fileList[mode]
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
                frame = self.prepareImg(frame)[0]
                batchX[bIdx,tIdx] = frame
            with self.pregraph.as_default():
                batchY = self.premodel.predict(np.reshape(batchX,(-1,self.sizeY,self.sizeX,self.nColor)))
            batchY = np.reshape(batchY,[self.nBatch, self.nLength] + [int(x) for x in batchY.shape[1:]])
            """
            testX  = batchX[0,-1]
            testY1 = batchY[0,-1,:,:,:self.np_branch1]
            testY2 = batchY[0,-1,:,:,self.np_branch1:]
            #print test.min(),test.max()
            img = self.showImg(testX,testY1,testY2)
            cv2.imshow("test",img)
            cv2.waitKey(1)
            """
            mov.release()
            bIdx += 1
        return {"inX":batchY},{"cls":batchT}

    def yieldOne(self,mode="all"):
        while True:
            yield self.loadOne(mode)

    def reloadModel(self,fPath):
        self.model.load_weights(fPath)
        print "model loaded from %s"%fPath
        return

    def loadModel(self,fPath):
        self.model = load_model(fPath)
        print "model loaded from %s"%fPath

    def buildPreModel(self):
        self.np_branch1 = np_branch1 = 38
        self.np_branch2 = np_branch2 = 19

        img_input = Input(shape=(self.sizeY, self.sizeX, self.nColor))
        stages = 6

        # VGG
        stage0_out = vgg_block(img_input)

        # stage 1
        stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1)
        stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2)
        x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

        # stage t >= 2
        for sn in range(2, stages + 1):
            stageT_branch1_out = stageT_block(x, np_branch1, sn, 1)
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2)
            if (sn < stages):
                x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

        x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])
        self.premodel = Model(inputs=img_input, outputs=x)
        self.premodel.summary()

        print "pre-trained parameter being loaded from %s"%self.weights_path
        self.premodel.load_weights(self.weights_path)
        print "...done"
        self.pregraph = tf.get_default_graph()

    def buildModel(self):
        input_shape = [self.nLength,] + [int(x) for x in self.premodel.output.shape[1:]]

        inX = Input(shape=input_shape,name="inX")
        h = inX
        h = Conv3D( 64,(5,3,3),activation="relu",padding="same")(h)
        h = Conv3D( 64,(5,3,3),activation="relu",padding="same")(h)
        h = Conv3D( 96,(5,3,3),activation="relu",padding="same")(h)
        h = Conv3D( 96,(5,3,3),activation="relu",padding="same")(h)
        h = MaxPooling3D(pool_size=(2,2,2),padding="same")(h)
        h = BatchNormalization(axis=-1)(h)
        h = Conv3D(128,(5,3,3), activation="relu",padding="same")(h)
        h = Conv3D(128,(5,3,3), activation="relu",padding="same")(h)
        h = MaxPooling3D(pool_size=(2,2,2),padding="same")(h)
        h = Conv3D(256,(5,3,3), activation="relu",padding="same")(h)
        h = Conv3D(256,(5,3,3), activation="relu",padding="same")(h)
        h = MaxPooling3D(pool_size=(2,2,2),padding="same")(h)
        h = Conv3D(self.nActivities,(5,3,3), activation="relu",padding="same")(h)
         
        h = TimeDistributed(GlobalMaxPooling2D())(h)
        h = GlobalAveragePooling1D()(h)
        logits = h
        output      = Activation("softmax",name="cls")(logits)

        model = Model(inputs=inX, outputs=output)

        model.compile(loss="categorical_crossentropy",optimizer=Adam(self.learnRate),metrics=["accuracy"])

        model.summary()
        self.model = model
        self.graph = tf.get_default_graph()

        return

    def train(self):
        cp_cb = ModelCheckpoint(filepath = self.saveFolder+"/weights.{epoch:02d}.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto')
        tb_cb = TensorBoard(log_dir=self.saveFolder, histogram_freq=1)
        lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-10, epsilon=0., verbose=1)

        self.model.fit_generator(generator=self.yieldOne("train"),
                                epochs=100000000,
                                callbacks=[cp_cb,tb_cb,lr_cb],
                                validation_data = self.yieldOne("test"),
                                validation_steps = 5,
                                steps_per_epoch=100,
                                use_multiprocessing=False)

    def test(self,movPath,outPath=None):
        actDict = self.actDict
        if movPath=="0":
            movFileList = [0]
        else:
            movFileList = glob.glob(movPath)

        if outPath:
            fourcc = cv2.VideoWriter_fourcc(*"H264")
            out = cv2.VideoWriter(outPath,fourcc, 30.0, (self.sizeX*3*2,self.sizeY*3))
        for movFile in movFileList:
            mov = cv2.VideoCapture(movFile)
            #inX = np.zeros((1,self.nLength,))
            inX = np.zeros([1,self.nLength,] + [int(x) for x in self.premodel.output.shape[1:]])
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
                oriFrame = frame.copy() # ここは元のものを保存
                frame = self.prepareImg(frame)[0]
                with self.pregraph.as_default():
                    processed = self.premodel.predict(np.expand_dims(frame,axis=0))
                #print frame.shape
                inX[0,:-1] = inX[0,-1:]
                inX[0,-1]  = processed
                ######inX = preprocess_input(inX)
                t   = self.model.predict(x=inX,batch_size=1)
                img = processed[0]
                map1 = img[:,:,:self.np_branch1]
                map2 = img[:,:,self.np_branch1:self.np_branch1+self.np_branch2]
                try:
                    poseImg = self.showImg(oriFrame,map1,map2)
                except:
                    print "error occured"
                v = t[0]
                t = self.cvtCls2act(np.argmax(t[0]))
                print t,v
                frame   = cv2.resize(oriFrame,(self.sizeX*3,self.sizeY*3))
                poseImg = cv2.resize(poseImg ,(self.sizeX*3,self.sizeY*3))
                #cv2.putText(  frame,t,(10,30),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 2)
                cv2.putText(poseImg,t,(10,30),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 2)
                #cv2.imshow("frame",frame)
                #cv2.imshow("pose",poseImg)
                bothImg = cv2.hconcat([frame,poseImg])
                cv2.imshow("both",bothImg)
                if outPath: out.write(bothImg)
                cv2.waitKey(int(1000/fps))
            mov.release()
        if outPath:
            out.release()


    def showImg(self,oriImg,map1,map2):
        #oriImg = self.prepareImg(oriImg)
        # extract outputs, resize, and remove padding
        #print map2.shape
        param, model_params = config_reader()
        imageToTest_padded, pad = util.padRightDownCorner(oriImg, model_params['stride'], model_params['padValue'])        
        heatmap = np.squeeze(map2) # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        #print heatmap.shape
        #print " oriImg:",oriImg.max(),oriImg.min()
        #print "heatmap:",heatmap.max(),heatmap.min()
        
        paf = np.squeeze(map1) # output 0 is PAFs
        paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        #print "    paf:",paf.max(),paf.min()

        from scipy.ndimage.filters import gaussian_filter
        all_peaks = []
        peak_counter = 0

        for part in range(19-1):
            map_ori = heatmap[:,:,part]
            map = gaussian_filter(map_ori, sigma=3)
            
            map_left = np.zeros(map.shape)
            map_left[1:,:] = map[:-1,:]
            map_right = np.zeros(map.shape)
            map_right[:-1,:] = map[1:,:]
            map_up = np.zeros(map.shape)
            map_up[:,1:] = map[:,:-1]
            map_down = np.zeros(map.shape)
            map_down[:,:-1] = map[:,1:]
            
            peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
            peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)


# find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],            [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],            [1,16], [16,18], [3,17], [6,18]]
# the middle joints heatmap correpondence
        mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22],           [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52],           [55,56], [37,38], [45,46]]


# In[19]:

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf[:,:,[x-19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0]-1]
            candB = all_peaks[limbSeq[k][1]-1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if(nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                        vec = np.divide(vec, norm)
                        
                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),                                np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                        
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]                                   for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]                                   for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                        criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0,5))
                for c in range(len(connection_candidate)):
                    i,j,s = connection_candidate[c][0:3]
                    if(i not in connection[:,3] and j not in connection[:,4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if(len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])


# last number in each row is the total parts number of that person
# the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:,0]
                partBs = connection_all[k][:,1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])): #= 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)): #1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1
                    
                    if found == 1:
                        j = subset_idx[0]
                        if(subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        print ("found = 2")
                        membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: #merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

# delete some rows of subset which has few parts occur
        deleteIdx = [];
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)


# visualize
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        cmap = matplotlib.cm.get_cmap('hsv')
        canvas = oriImg
        stickwidth = 4

# Link body parts

        ret = []

        for i in range(17):
            #for n in range(len(subset)):
            if len(subset)==0:
                ret.append([i,0.,0.])
                print "could not find human"
                continue
            n = 0 # 写っている人物は1人だけに限定する
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            #print i,n,mX,mY
            ret.append([i,mX,mY])
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        return canvas
        #plt.imshow(canvas[:,:,[2,1,0]])
        """
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(12, 12)
        plt.show()
        raw_input()
        """
        #return ret, canvas[:,:,[2,1,0]]


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nBatch" ,"-b",dest="nBatch",type=int,default=8)
    parser.add_argument("--nLength","-l",dest="nLength"  ,type=int,default=30)
    parser.add_argument("--learnRate",dest="learnRate"  ,type=float,default=1e-4)
    parser.add_argument("--doFineTune","-f",dest="doFineTune"  ,action="store_true")
    parser.add_argument("--reload","-r",dest="reload"  ,type=str,default=None)
    parser.add_argument("--saveFolder","-s",dest="saveFolder"  ,type=str,default="save")
    parser.add_argument("--test","-t",dest="test"  ,type=str,default=None)
    parser.add_argument("--testSave",dest="testSave"  ,type=str,default=None)
    args = parser.parse_args()

    if args.test:
        assert args.reload, "please set model to use"
        n = net(nBatch=1,
                nLength=args.nLength,
                learnRate=0,
                isTraining=False,
                saveFolder=None)
        n.buildPreModel()
        n.buildModel()
        if args.reload:
            n.reloadModel(args.reload)
        n.test(args.test,args.testSave)
    else:
        n = net(nBatch=args.nBatch,
                nLength=args.nLength,
                learnRate=args.learnRate,
                saveFolder=args.saveFolder)
        n.buildPreModel()
        n.buildModel()
        if args.reload:
            n.reloadModel(args.reload)
        n.train()
