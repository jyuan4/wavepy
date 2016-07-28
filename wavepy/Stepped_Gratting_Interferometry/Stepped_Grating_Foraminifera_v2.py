'''
Stepped_Grating_Forminifera.nb
29 July 2014

Sixteen stepped positions with a period of 10.
4.8 \[Mu]m linear absorption and phase gratings. __ keV x-ray energy 0.58 sec exposure time.  1st Talbot distance.  
Pixel size = 6.48/5=1.29 \[Micro]m
'''

# -*- coding: utf-8 -*-
from __future__ import print_function


#%%
'''*********************************************************
Step 1:  Get Filenames of white, dark, and sample tiff files: 
'''
import os
from os import listdir
from os.path import isfile, join
import numpy  
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

notebookDirectoryPath = '/Users/jumaoyuan/Downloads/tomopy_Jumao' #change directory here
pathData = os.path.join(notebookDirectoryPath, 'sample/') 
pathReference = os.path.join(notebookDirectoryPath, 'reference/') 
pathDark = os.path.join(notebookDirectoryPath, 'dark/')
pathProjections = notebookDirectoryPath
pathFigures = os.path.join(notebookDirectoryPath, 'figures/') 
pathFITS = os.path.join(notebookDirectoryPath, 'FITS/')
nameSample = "12Mar_1T_58mm_foram_"
nameStr = "foram_"

def readFile(path):
    numFiles = [f for f in listdir(path) if isfile(join(path,f))]
    images = numpy.empty(len(numFiles), dtype=object)
    print ('There are '+ str(len(numFiles))+ ' files')  
    I = Image.open(os.path.join(path, numFiles[0]))
    I_array = numpy.array(I)
    I_array = np.zeros((len(numFiles),I_array.shape[0], I_array.shape[1]))   
    for n in range(0, len(numFiles)):
        numFiles[n] = join(path,numFiles[n])
        images[n] = Image.open(numFiles[n])
        I_array[n,:,:]= numpy.array(images[n])
    return(numFiles)

filenamesSample = readFile(pathData) 
filenamesReference = readFile(pathReference)   
filenamesDark = readFile(pathDark)


#%%
pixelSize = 6.48/10 # micron #
print (str(len(filenamesSample)) + '  ' + str(len(filenamesReference)) + ' ' + str(len(filenamesDark)))
filenamesSample[0]
filenamesSample[len(filenamesSample)-1]
plt.imshow(Image.open(filenamesSample[1]))
plt.show()
[rowsOriginal, columnsOriginal] = numpy.array(Image.open(filenamesSample[1])).shape

#%%
numberGratingSteps = 16
array_numberGratingSteps = np.zeros(numberGratingSteps)
for i in range(numberGratingSteps):
    array_numberGratingSteps[i] = i
listGratingStepsMicron = 0.48 * array_numberGratingSteps
print (listGratingStepsMicron)
gratingPeriodMicron = 4.8

#%%
def figFunc(climRange, data):
    fig1 = plt.gcf()
    plt.imshow(data, vmin = climRange[0], vmax = climRange[1])
    plt.colorbar()
    plt.show()     
    plt.draw()  
    return fig1
    
#%%
dataRaw = numpy.array(Image.open(filenamesSample[0]))
climRaw = [dataRaw.min(), dataRaw.max()]         
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
fig1 = figFunc(climRaw, dataRaw)       
#fig1.savefig(os.path.join(notebookDirectoryPath, 'foram_raw.png'), dpi=200)

#%% export sample dataraw figure
fig1 = plt.gcf()
plt.imshow(dataRaw, vmin = climRaw[0], vmax = climRaw[1], cmap = 'Greys_r')
#plt.title('Sample')
plt.colorbar()
plt.show()     
plt.draw()  
fig1.savefig(os.path.join(notebookDirectoryPath, 'foram_raw.png'), dpi=200)
    
#%%
'''*****************************************************************************
Step 2: Calculate average dark image variables:  dataDark, rows, columns, NY, NX
'''
dataDark = np.zeros((len(filenamesDark),numpy.array(Image.open(filenamesDark[1])).shape[0], numpy.array(Image.open(filenamesDark[1])).shape[1]))
for i in range(len(filenamesDark)):
    dataDark[i,:,:] = numpy.array(Image.open(filenamesDark[i]))
print ('The dimension of dataDark is ' + str(dataDark.shape))
print (str(dataDark.min()) + '   ' + str(dataDark.max()) + '   ' + str(dataDark.shape))
#[x, y] = dataDark[1,:,:].shape
dataDark = np.median(dataDark, axis = 0)
print (dataDark.shape)
print (str(dataDark.min())  + '   '  + str(dataDark.max()) + '  '  + str(dataDark.shape))

#%%
import pylab
[rows, columns] = [NY, NX] = dataDark.shape
fig0 = pylab.figure()
fig1 = fig0.add_subplot(2,1,1)
fig1 = plt.gcf()
plt.imshow(dataDark, cmap='Greys_r')
fig2 = fig0.add_subplot(2,2,1)
fig2 = plt.hist(dataDark.flatten(),100, facecolor='g', alpha=0.75)

#%%
'''*****************************************************************************
Step 3: Define functions:  funcPrepareBvector.  Subscript[p, g]=10
'''
import math
def funcPrepareBvectorArbitrarySteps(gratingPeriodMicron, listGratingStepsMicron):
    numberGratingSteps = len(listGratingStepsMicron)
    b1 = np.ones((numberGratingSteps,1), dtype = np.int).flatten()
    b2 = np.sin(2 * math.pi * listGratingStepsMicron / gratingPeriodMicron)
    b3 = np.cos(2 * math.pi * listGratingStepsMicron / gratingPeriodMicron)
    return np.round(np.transpose(numpy.vstack([b1,b2,b3])),6)
bVector = funcPrepareBvectorArbitrarySteps(gratingPeriodMicron, listGratingStepsMicron)
np.matrix(bVector)

#%%
from numpy.linalg import inv  
aVector = cVector = np.zeros((3, rows * columns))
aMatrix = np.zeros((rows, columns, 3))
visibility = phi = np.zeros((rows, columns))
gMatrix = np.dot(inv(np.dot(np.transpose(bVector), bVector)), np.transpose(bVector))
print (gMatrix.shape)
print (np.matrix(gMatrix))

#%%
'''*****************************************************************************
Step 4:  For reference images, calculate coefficients: refTransmission, refPhi, refVisibility
'''   

def oneImageOneData(fname, dataDark):
    allData = np.zeros((rows, columns, numberGratingSteps))
    for k in range(numberGratingSteps):
        oneImage = Image.open(fname[k])
        oneData = np.array(oneImage) - dataDark
        allData[:,:,k] = oneData
    return allData   
allReferenceData = oneImageOneData(filenamesReference, dataDark)
print (allReferenceData.shape)

#%%
cVector = np.transpose(np.reshape(allReferenceData, (rows*columns, numberGratingSteps)))
print (cVector.shape)
aVector = np.dot(gMatrix, cVector)
aMatrix = np.reshape(np.transpose(aVector), (rows, columns, 3)) 
refVisibility = np.sqrt(aMatrix[:,:,1]**2 + aMatrix[:,:,2]**2)
refPhi = np.arctan2(aMatrix[:,:,2], aMatrix[:,:,1])
refTransmission = aMatrix[:,:,0]
refVisibilityPercent = 100 * refVisibility / refTransmission
print (str(refTransmission.shape) + '  ' + str(refPhi.shape) + '   ' + str(refVisibility.shape))

#%%
'''*****************************************************************************
Step 4.1:  (Optional) For reference images, calculate chiSquare of the sinusoidal fit (about 0.5 minutes)
'''
import time
print (np.transpose(np.reshape(aMatrix, (rows * columns, 3))).shape)
print (bVector.shape)
print (cVector.shape)
countsCalculated = np.dot(bVector, np.transpose(np.reshape(aMatrix, (rows * columns, 3))))
print (countsCalculated.shape)

#%%
def chiSquare(cVector, countsCalculated):
    data = np.zeros(rows * columns)
    print (data.shape)
    for p in range(cVector.shape[1]):
        cg = cVector[:, p]
        cgHat = countsCalculated[:, p]
        data[p] = np.sum((cg - cgHat)**2 / cg) / (numberGratingSteps - 3 - 1)  
    return data

start = time.time()    
chiSquareRef = chiSquare(cVector, countsCalculated)
end = time.time()
print ('The elasped time is ' + str(end - start) + ' seconds')
new_chiSquareRef = np.reshape(chiSquareRef, (rows, columns))
print (new_chiSquareRef.shape)
chiSquareRef = new_chiSquareRef

#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
climChiSquare = ([0, 5])
textStr =  str(round(float(listGratingStepsMicron[-1]),2)).replace('.','p')
fig1 = figFunc(climChiSquare, chiSquareRef)
fig1.savefig(os.path.join(notebookDirectoryPath, nameStr +  'chisquareRef_' + textStr + '_1.png'), dpi=200)
fig2 = plt.hist(chiSquareRef.flatten(),30, range = [0, 5], facecolor='b', alpha=0.75) # alpha range 0-1 #
plt.xlabel('$\chi_v^2$', fontsize = 18)
plt.ylabel('counts', fontsize = 18)
plt.savefig(os.path.join(notebookDirectoryPath, nameStr +  'chisquareRef_' + textStr + '_2.png'), dpi=200, bbox_inches='tight', pad_inches=0)

#%%
'''*****************************************************************************
Step 5:  For sample images, calculate coefficients: sampleTransmission, samplePhi, sampleVisibility
'''
allSampleData = oneImageOneData(filenamesSample, dataDark)
print (allSampleData.shape)
print (aVector.shape)
print (aMatrix.shape)

#%%
cVector = np.transpose(np.reshape(allSampleData, (rows*columns, numberGratingSteps)))#flatten to 2d#
print (cVector.shape)
aVector = np.dot(gMatrix, cVector)
aMatrix = np.reshape(np.transpose(aVector), (rows, columns, 3)) #partition = np.split#
sampleVisibility = np.sqrt(aMatrix[:,:,1]**2 + aMatrix[:,:,2]**2)
samplePhi = np.arctan2(aMatrix[:,:,2], aMatrix[:,:,1])
sampleTransmission = aMatrix[:,:,0]
sampleVisibilityPercent = 100 * sampleVisibility / sampleTransmission
print (str(sampleTransmission.shape) + '   ' + str(samplePhi.shape) + '  ' + str(sampleVisibility.shape))

#%%
climTrans = ([sampleTransmission.min(), 0.6 * sampleTransmission.max()])
fig1 = figFunc(climTrans, sampleTransmission) 
fig1.savefig(os.path.join(notebookDirectoryPath, nameStr +  'trans' + '.png'), dpi=200)

climVis = ([sampleVisibility.min(), 0.5 * sampleVisibility.max()])
fig1 = figFunc(climVis, sampleVisibility) 
fig1.savefig(os.path.join(notebookDirectoryPath, nameStr +  'vis' + '.png'), dpi=200)

#%%
climVisPercent = ([0, 40])
fig1 = plt.gcf()
plt.imshow(sampleVisibilityPercent, vmin = climVisPercent[0], vmax = climVisPercent[1], cmap = 'Greys_r')
#plt.title('Sample')
plt.colorbar()
plt.show()     
plt.draw() 
textStr =  str(round(float(listGratingStepsMicron[-1]),2)).replace('.','p')
fig1.savefig(os.path.join(notebookDirectoryPath, nameStr +  'visPercent_' + textStr +  '_1.png'), dpi=200)

fig2 = plt.hist(sampleVisibilityPercent.flatten(),30, range = [0, 50], facecolor='b', alpha=0.75)
plt.xlabel('% visibility',fontsize = 18)
plt.ylabel('counts', fontsize = 18)
plt.savefig(os.path.join(notebookDirectoryPath, nameStr +  'visPercent_' + textStr +  '_2.png'), dpi=200, bbox_inches='tight', pad_inches=0)

#%%
climPhi = ([samplePhi.min(), samplePhi.max()])
fig1 = figFunc(climPhi, samplePhi)
#fig1.savefig(os.path.join(notebookDirectoryPath, nameStr +  'vphi' + '.png'), dpi=200)

#%%
import scipy
import scipy.ndimage
import scipy.signal
import pylab as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
smallSamplePhi = scipy.ndimage.interpolation.zoom(scipy.signal.medfilt(samplePhi, 1), 1/math.ceil(columns/200))
print (smallSamplePhi.dtype)
fig1 = figFunc(climPhi, smallSamplePhi)      
fig1.savefig(os.path.join(notebookDirectoryPath, 'foram_phi_1.png'), dpi=200)

#%%
smallSamplePhi2 = np.resize(smallSamplePhi, (rows, columns))
#%% 3D image  
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=plt.figaspect(5.0/7))
ax = fig.add_subplot(111, projection='3d')

xxGrid, yyGrid = np.meshgrid(np.arange(smallSamplePhi2.shape[0]),
                             np.arange(smallSamplePhi2.shape[1]),
                             indexing='ij')

surf = ax.plot_surface(xxGrid, yyGrid, smallSamplePhi2,
                rstride=1, cstride=1,
                cmap='spectral', linewidth=0)

#plt.xlabel('rows', fontsize=16)
#plt.ylabel('columns', fontsize=16)
plt.xlim([0, rows])
plt.ylim([0, columns])
ax.set_xlabel('rows')
ax.set_ylabel('columns')
ax.set_zlabel('phi')
#    ax.set_zlim(0, .50)
#plt.title('smallSamplePhi', fontsize=18, weight='bold')
#plt.colorbar(surf)


plt.show(block=True)
fig.savefig(os.path.join(notebookDirectoryPath, 'foram_phi_2.png'), dpi=200)
#%%
'''*****************************************************************************
Step 6: (Optional) For sample images, calculate chiSquare of the sinusoidal fit (about 0.5 minutes)
'''
print (np.transpose(np.reshape(aMatrix, (rows * columns, 3))).shape)
print (bVector.shape)
print (cVector.shape)
countsCalculated = np.dot(bVector, np.transpose(np.reshape(aMatrix, (rows * columns, 3))))
print (countsCalculated.shape)

#%%
start = time.time() 
chiSquareSample = chiSquare(cVector, countsCalculated)
end = time.time()
print ('The elasped time is ' + str(end - start) + ' seconds')
new_chiSquareSample =np.reshape(chiSquareSample, (rows, columns))
print (new_chiSquareSample.shape)
chiSquareSample = new_chiSquareSample

#%%
import matplotlib.pyplot as plt
import numpy as np
climChiSquare = ([0, 5])
fig1 = figFunc(climChiSquare, chiSquareSample)
textStr =  str(round(float(listGratingStepsMicron[-1]),2)).replace('.','p')
fig1.savefig(os.path.join(notebookDirectoryPath, nameStr +  'chisquareSample_' + textStr + '_1.png'), dpi=200)
fig2 = plt.hist(chiSquareSample.flatten(),30, range = [0, 6], facecolor='b', alpha=0.75)
plt.xlabel('$\chi_v^2$', fontsize = 18)
plt.ylabel('counts', fontsize = 18)
plt.savefig(os.path.join(notebookDirectoryPath, nameStr +  'chisquareSample_' + textStr + '_2.png'), dpi=200, bbox_inches='tight', pad_inches=0)

#%%
'''*****************************************************************************
Step 6.1:  (Optional) For one point, calculate best fit with NLM and linear algebra
'''
'''----------------------------------------------
(1) get the reference and sample data for a test point
'''

print (listGratingStepsMicron)
testPoint = np.int_(numpy.round((rows/2, columns/2)) + (0,0))
print (testPoint)
print (sampleTransmission[testPoint[0]-1, testPoint[1]-1])
print (sampleVisibility[testPoint[0]-1, testPoint[1]-1])
print (samplePhi[testPoint[0]-1, testPoint[1]-1])
listDataReference = allReferenceData[testPoint[0]-1, testPoint[1]-1, :]
listDataSample = allSampleData[testPoint[0]-1, testPoint[1]-1, :]
print (listDataReference)
print (listDataSample)

#%%
'''----------------------------------------------
(2) Fit with NLM
'''
import numpy as np
from scipy.optimize import minimize
def NonLinearModelFit(x, *p):
    a, b, c = p
    return a + b * np.sin(x * (2 * math.pi / gratingPeriodMicron) + c)
    
Ref_tmp = np.transpose(np.concatenate((np.matrix(listGratingStepsMicron), np.matrix(listDataReference)), axis=0))
Sample_tmp = np.transpose(np.concatenate((np.matrix(listGratingStepsMicron), np.matrix(listDataSample)), axis=0)) 
X_ref = np.squeeze(np.asarray(Ref_tmp[:,0]))
Y_ref =np.squeeze(np.asarray(Ref_tmp[:,1]))
X_sample = np.squeeze(np.asarray(Sample_tmp[:,0]))
Y_sample =np.squeeze(np.asarray(Sample_tmp[:,1]))

#%%
def fitNLM(p0, x, y_noise, p_init):
    err = lambda p: np.mean((NonLinearModelFit(x, *p)-y_noise)**2)
    p_opt = minimize(
                 err,
                 p_init,
                 bounds = [(None, None), (0, None), (-math.pi, math.pi)],
                 method="L-BFGS-B"
                 ).x    
    return p_opt                 
p_opt_ref = fitNLM([1000, 200, 0.7], X_ref, Y_ref, [1000, 200, 0])
plt.scatter(X_ref, Y_ref, alpha=.2, c='b', label="f + noise")
plt.plot(X_ref, NonLinearModelFit(X_ref, *p_opt_ref), '--', c='r', lw=2., label="fitted f")                
residuals = Y_ref - NonLinearModelFit(X_ref, *p_opt_ref)
chiSquareRefOnePoint = round(float(np.sum(residuals**2/listDataReference)/(len(listGratingStepsMicron)-3-1)), 2)
print (chiSquareRefOnePoint)

#%%         
p_opt_sample = fitNLM([1000, 200, 0.7], X_sample, Y_sample, [1000, 200, 0])
plt.scatter(X_sample, Y_sample, alpha=.2, c='b', label="f + noise")
plt.plot(X_sample, NonLinearModelFit(X_sample, *p_opt_sample), '--', c='r', lw=2., label="fitted f")                
residuals = Y_sample - NonLinearModelFit(X_sample, *p_opt_sample)
chiSquareSampleOnePoint = round(float(np.sum(residuals**2/listDataSample)/(len(listGratingStepsMicron)-3-1)), 2)
print(chiSquareSampleOnePoint)

#%%
fig1 = plt.gcf()
plt.scatter(X_ref, Y_ref, alpha=.2, c='b', label="f + noise")
plt.scatter(X_sample, Y_sample, alpha=.2, c='b', label="f + noise")
plt.plot(X_ref, NonLinearModelFit(X_ref, *p_opt_ref), '--', c='r', lw=2., label="fitted f")  
plt.text(1.5, 1700, r'$X_v^2(ref)=0.76$', color = 'red')
plt.plot(X_sample, NonLinearModelFit(X_sample, *p_opt_sample), '--', c='b', lw=2., label="fitted f")   
plt.text(1.5, 1650, r'$X_v^2(sample)=1.29$', color = 'blue')
plt.xlabel('$X_g$,micros', fontsize = 18) 
plt.ylabel('counts', fontsize = 18)
plt.show()
fig1.savefig(os.path.join(notebookDirectoryPath,'foram_NLM_fit_and_parameters.png'), dpi=200, bbox_inches='tight', pad_inches=0)
#%%
'''----------------------------------------------
(3) Fit with Linear Algebra
'''
listGratingStepsMicron = np.transpose(listGratingStepsMicron).flatten()
print (listGratingStepsMicron)
b1 = np.ones((len(listGratingStepsMicron),1), dtype = np.int).flatten()# non-float #
b2 = np.zeros(len(listGratingStepsMicron)).flatten()
for xg in range(len(listGratingStepsMicron)):
    b2[xg] = np.sin(listGratingStepsMicron[xg] * (2 * math.pi / gratingPeriodMicron))
b3 = np.zeros(len(listGratingStepsMicron))
for xg in range(len(listGratingStepsMicron)):
    b3[xg] = np.cos(listGratingStepsMicron[xg] * (2 * math.pi / gratingPeriodMicron))
bVector = np.round(np.vstack((b1,b2,b3)),6) #chop matrix in mathematcia#
bVector = np.transpose(bVector)
print (bVector.shape)
from numpy.linalg import inv
gMatrix = np.dot(inv(np.dot(np.transpose(bVector), bVector)), np.transpose(bVector))
print (np.matrix(gMatrix))
print (gMatrix.shape)

#%%
fitByVectorizedLinearAlgebra = np.dot(gMatrix, listDataReference)
laTransmissionRef = fitByVectorizedLinearAlgebra[0]
laAmplitudeRef = np.sqrt(fitByVectorizedLinearAlgebra[1]**2 + fitByVectorizedLinearAlgebra[2]**2)
laPhiRef = np.arctan2(fitByVectorizedLinearAlgebra[2] , fitByVectorizedLinearAlgebra[1])
listFitLinearAlgebra = np.zeros(len(listGratingStepsMicron))
for xg in range(len(listGratingStepsMicron)):
    listFitLinearAlgebra[xg] = laTransmissionRef + laAmplitudeRef * np.sin(listGratingStepsMicron[xg]* \
(2 * math.pi / gratingPeriodMicron) + laPhiRef)
residuals = np.zeros(len(listGratingStepsMicron))
for i in range(len(listGratingStepsMicron)):
    residuals[i] = listDataReference[i] - listFitLinearAlgebra[i]
chiSquareRefOnePoint = np.round_(np.sum(residuals**2/listDataReference) / (len(listGratingStepsMicron) - 3 - 1), 2)
print (chiSquareRefOnePoint)

#%%
fitByVectorizedLinearAlgebra = np.dot(gMatrix, listDataSample)
laTransmissionSample = fitByVectorizedLinearAlgebra[0]
laAmplitudeSample = np.sqrt(fitByVectorizedLinearAlgebra[1]**2 + fitByVectorizedLinearAlgebra[2]**2)
laPhiSample = np.arctan2(fitByVectorizedLinearAlgebra[2] , fitByVectorizedLinearAlgebra[1])
listFitLinearAlgebra = np.zeros(len(listGratingStepsMicron))
for xg in range(len(listGratingStepsMicron)):
    listFitLinearAlgebra[xg] = laTransmissionSample + laAmplitudeSample * np.sin(listGratingStepsMicron[xg]* \
(2 * math.pi / gratingPeriodMicron) + laPhiSample)
residuals = np.zeros(len(listGratingStepsMicron))
for i in range(len(listGratingStepsMicron)):
    residuals[i] = listDataSample[i] - listFitLinearAlgebra[i]
chiSquareSampleOnePoint = np.round_(np.sum(residuals**2/listDataSample) / (len(listGratingStepsMicron) - 3 - 1), 2)
print (chiSquareSampleOnePoint)
    

#%%
plt.scatter(X_ref, Y_ref, alpha=.2, c='b', label="f + noise")
plt.scatter(X_sample, Y_sample, alpha=.2, c='b', label="f + noise")
plt.plot(X_ref, laTransmissionRef+laAmplitudeRef*np.sin(listGratingStepsMicron*(2*math.pi/gratingPeriodMicron)+laPhiRef\
                                                        ), '--', c='r', lw=2., label="fitted f") 
plt.plot(X_sample, laTransmissionSample+laAmplitudeSample*np.sin(listGratingStepsMicron*(2*math.pi/gratingPeriodMicron)+laPhiSample\
                                                           ),'--', c='b', lw=2., label="fitted f") 
plt.show()


#%%
absorption = -np.log(laTransmissionSample / laTransmissionRef)
differentialPhase = laPhiSample - laPhiRef
percentVisibility = 100 * laAmplitudeSample / laTransmissionSample
darkfield = (laAmplitudeSample / laAmplitudeRef) / (laTransmissionSample / laTransmissionRef)
print (absorption)
print(differentialPhase)
print (percentVisibility)
print (darkfield)

#%%
'''*****************************************************************************
Step 7:  Absorption: calculated from sampleTransmission and refTransmission
'''
absorption = -np.log(sampleTransmission / refTransmission)
print (str(absorption.min()) + '   ' + str(np.mean(absorption.flatten())) + '    ' + str(absorption.max()))
climAbs = ([-0.1, 1])
fig1 = figFunc(climAbs, absorption)
fig1.savefig(os.path.join(notebookDirectoryPath, nameStr + 'absorption.png'), dpi=200)

#%%
'''*****************************************************************************
Step 8: Differential Phase Contrast: calculated from samplePhi and refPhi
'''
differentialPhase = samplePhi - refPhi
print (str(differentialPhase.min()) + '   ' +  str(np.mean(differentialPhase.flatten())) + '   '+ \
           str(differentialPhase.max()))
smallDPC =  scipy.ndimage.interpolation.zoom(scipy.signal.medfilt(differentialPhase, 1), 1/math.ceil(columns/200))
climADPC = ([-0.2, 1])
fig1 = figFunc(climADPC, differentialPhase)
fig1.savefig(os.path.join(notebookDirectoryPath, nameStr + 'DPC_1.png'), dpi=200)
fig2 = plt.hist(differentialPhase.flatten(), 100, range=[-math.pi, math.pi], facecolor='b', alpha=0.75) # alpha range 0-1 #
plt.savefig(os.path.join(notebookDirectoryPath, nameStr + 'DPC_2.png'), dpi=200, bbox_inches='tight', pad_inches=0)
#%% 3D image  

fig = plt.figure(figsize=plt.figaspect(5.0/7))
ax = fig.add_subplot(111, projection='3d')

xxGrid, yyGrid = np.meshgrid(np.arange(smallDPC.shape[0]),
                             np.arange(smallDPC.shape[1]),
                             indexing='ij')

surf = ax.plot_surface(xxGrid, yyGrid, smallDPC,
                rstride=1, cstride=1,
                cmap='spectral', linewidth=0)
#cmap options: http://mpastell.com/2013/05/02/matplotlib_colormaps/

#plt.xlabel('rows', fontsize=16)
#plt.ylabel('columns', fontsize=16)
ax.set_xlabel('rows')
ax.set_ylabel('columns')
ax.set_zlabel('phi')
#    ax.set_zlim(0, .50)
#plt.title('smallDPC', fontsize=18, weight='bold')
#plt.colorbar(surf)


plt.show(block=True)
fig.savefig(os.path.join(notebookDirectoryPath, nameStr + 'DPC_3.png'), dpi=200)
#%%
'''*****************************************************************************
Step 9: Dark-Field: calculated from sampleVisibility and refVisibility and normalized with transmission
'''
darkfield = (sampleVisibility / refVisibility) / (sampleTransmission / refTransmission)
print (str(darkfield.min()) + '   ' +  str(np.mean(darkfield.flatten())) + '   '+ \
           str(darkfield.max()))
climDarkfield = ([-0.1, 1])
fig1 = figFunc(climDarkfield, darkfield)
fig1.savefig(os.path.join(notebookDirectoryPath, nameStr + 'darkfield_1.png'), dpi=200)
fig2 = plt.hist(darkfield.flatten(), 100, range=[0, 1.5], facecolor='b', alpha=0.75) # alpha range 0-1 #
plt.savefig(os.path.join(notebookDirectoryPath, nameStr + 'darkfield_2.png'), dpi=200, bbox_inches='tight', pad_inches=0)

#%%
'''*****************************************************************************
Step 9.1: Plots of upper left corner (paper fig.11)
'''
cropLimitRows = ([1-1, 100])
cropLimitColumns = ([1-1, 150])
'''-------------------------------------
transmission
'''
cropSampleTransmission  = sampleTransmission[cropLimitRows[0]:cropLimitRows[1], cropLimitColumns[0]:cropLimitColumns[1]]
climTrans = ([cropSampleTransmission.min(), 1 * cropSampleTransmission.max()])
fig1 = figFunc(climTrans, cropSampleTransmission)
fig1.savefig(os.path.join(notebookDirectoryPath, nameStr + 'trans_1.png'), dpi=200)

cropRefTransmission = refTransmission[cropLimitRows[0]:cropLimitRows[1], cropLimitColumns[0]:cropLimitColumns[1]]
fig1 = figFunc(climTrans, cropRefTransmission)
fig1.savefig(os.path.join(notebookDirectoryPath, nameStr + 'trans_2.png'), dpi=200)

climT = ([-100, 100])
fig1 = figFunc(climT, cropSampleTransmission - cropRefTransmission)

#%% 3D image
cropDiff = cropSampleTransmission - cropRefTransmission
fig = plt.figure(figsize=plt.figaspect(5.0/7))
ax = fig.add_subplot(111, projection='3d')

xxGrid, yyGrid = np.meshgrid(np.arange(cropDiff.shape[0]),
                             np.arange(cropDiff.shape[1]),
                             indexing='ij')

surf = ax.plot_surface(xxGrid, yyGrid, cropDiff,
                rstride=1, cstride=1,
                cmap='spectral', linewidth=0)
#cmap options: http://mpastell.com/2013/05/02/matplotlib_colormaps/

#plt.xlabel('rows', fontsize=16)
#plt.ylabel('columns', fontsize=16)
ax.set_xlabel('rows')
ax.set_ylabel('columns')
ax.set_zlabel('phi')
#    ax.set_zlim(0, .50)
#plt.title('cropSample-cropRef/Transmission', fontsize=18, weight='bold')
#plt.colorbar(surf)


plt.show(block=True)
fig.savefig(os.path.join(notebookDirectoryPath, nameStr + 'trans_3.png'), dpi=200)

#%%
'''-------------------------------------
absorption
'''
cropAbsorption =  absorption[cropLimitRows[0]:cropLimitRows[1], cropLimitColumns[0]:cropLimitColumns[1]]
climAbsHist = np.round_(0.6 * np.abs(climAbs).min(), 2)
climAbsHist = ([-climAbsHist, climAbsHist, 0.005])
climAbs = np.round_([cropAbsorption.min(), cropAbsorption.max()], 2)
fig1 = figFunc(climAbs, cropAbsorption)
fig1.savefig(os.path.join(notebookDirectoryPath, nameStr + 'abs_1.png'), dpi=200)
fig2 = plt.hist(cropAbsorption.flatten(), 30, range=[-0.06,0.06], facecolor='b', alpha=0.75) # alpha range 0-1 #
plt.savefig(os.path.join(notebookDirectoryPath, nameStr + 'abs_2.png'), dpi=200, bbox_inches='tight', pad_inches=0)
print (np.std(cropAbsorption.flatten()))

#%% 3D image
fig = plt.figure(figsize=plt.figaspect(5.0/7))
ax = fig.add_subplot(111, projection='3d')

xxGrid, yyGrid = np.meshgrid(np.arange(cropAbsorption.shape[0]),
                             np.arange(cropAbsorption.shape[1]),
                             indexing='ij')

surf = ax.plot_surface(xxGrid, yyGrid, cropAbsorption,
                rstride=1, cstride=1,
                cmap='spectral', linewidth=0)
#cmap options: http://mpastell.com/2013/05/02/matplotlib_colormaps/

#plt.xlabel('rows', fontsize=16)
#plt.ylabel('columns', fontsize=16)
ax.set_xlabel('rows')
ax.set_ylabel('columns')
ax.set_zlabel('phi')
#    ax.set_zlim(0, .50)
#plt.title('cropAbs', fontsize=18, weight='bold')
#plt.colorbar(surf)

plt.show(block=True)
fig.savefig(os.path.join(notebookDirectoryPath, nameStr + 'abs_3.png'), dpi=200)
#%%
'''-------------------------------------
phi
'''
cropSamplePhi = samplePhi[cropLimitRows[0]:cropLimitRows[1], cropLimitColumns[0]:cropLimitColumns[1]]
climPhi = ([cropSamplePhi.min(), 1 * cropSamplePhi.max()])
fig1 = figFunc(climPhi, cropSamplePhi) 
fig2 = plt.hist(cropSamplePhi.flatten(), 1000, facecolor='g', alpha=0.75) # alpha range 0-1 #

#%% 3D image
fig = plt.figure(figsize=plt.figaspect(4.0/8))
ax = fig.add_subplot(111, projection='3d')

xxGrid, yyGrid = np.meshgrid(np.arange(cropSamplePhi.shape[0]),
                             np.arange(cropSamplePhi.shape[1]),
                             indexing='ij')

surf = ax.plot_surface(xxGrid, yyGrid, cropSamplePhi,
                rstride=1, cstride=1,
                cmap='spectral', linewidth=0)
#cmap options: http://mpastell.com/2013/05/02/matplotlib_colormaps/

#plt.xlabel('rows', fontsize=16)
#plt.ylabel('columns', fontsize=16)
ax.set_xlabel('rows')
ax.set_ylabel('columns')
ax.set_zlabel('phi')
#    ax.set_zlim(0, .50)
plt.title('cropSamplePhi', fontsize=18, weight='bold')
#plt.colorbar(surf)

plt.show(block=True)
#%%
'''-------------------------------------
visibility
'''
cropSampleVisibilityPercent = sampleVisibilityPercent[cropLimitRows[0]:cropLimitRows[1], cropLimitColumns[0]:cropLimitColumns[1]]
climVisPercent = ([cropSampleVisibilityPercent.min(), 1*cropSampleVisibilityPercent.max()])
fig1 = figFunc(climVisPercent, cropSampleVisibilityPercent)
fig2 = plt.hist(cropSampleVisibilityPercent.flatten(), 1000, facecolor='g', alpha=0.75) # alpha range 0-1 #

#%% 3D image
fig = plt.figure(figsize=plt.figaspect(4.0/8))
ax = fig.add_subplot(111, projection='3d')

xxGrid, yyGrid = np.meshgrid(np.arange(cropSampleVisibilityPercent.shape[0]),
                             np.arange(cropSampleVisibilityPercent.shape[1]),
                             indexing='ij')

surf = ax.plot_surface(xxGrid, yyGrid, cropSampleVisibilityPercent,
                rstride=1, cstride=1,
                cmap='spectral', linewidth=0)
#cmap options: http://mpastell.com/2013/05/02/matplotlib_colormaps/

#plt.xlabel('rows', fontsize=16)
#plt.ylabel('columns', fontsize=16)
ax.set_xlabel('rows')
ax.set_ylabel('columns')
ax.set_zlabel('phi')
#    ax.set_zlim(0, .50)
plt.title('cropSampleVisibilityPercent', fontsize=18, weight='bold')
#plt.colorbar(surf)

plt.show(block=True)

#%%
'''-------------------------------------
chi-square
'''
cropChiSquareSample = chiSquareSample[cropLimitRows[0]:cropLimitRows[1], cropLimitColumns[0]:cropLimitColumns[1]]
climChiSquare = ([0, 5])
fig1 = figFunc(climChiSquare, cropChiSquareSample)
fig2 = plt.hist(cropChiSquareSample.flatten(), 1000, facecolor='g', alpha=0.75) # alpha range 0-1 

#%% 3D image
fig = plt.figure(figsize=plt.figaspect(4.0/8))
ax = fig.add_subplot(111, projection='3d')

xxGrid, yyGrid = np.meshgrid(np.arange(cropChiSquareSample.shape[0]),
                             np.arange(cropChiSquareSample.shape[1]),
                             indexing='ij')

surf = ax.plot_surface(xxGrid, yyGrid, cropChiSquareSample,
                rstride=1, cstride=1,
                cmap='spectral', linewidth=0)
#cmap options: http://mpastell.com/2013/05/02/matplotlib_colormaps/

#plt.xlabel('rows', fontsize=16)
#plt.ylabel('columns', fontsize=16)
ax.set_xlabel('rows')
ax.set_ylabel('columns')
ax.set_zlabel('phi')
#    ax.set_zlim(0, .50)
plt.title('cropChiSquareSample', fontsize=18, weight='bold')
#plt.colorbar(surf)

plt.show(block=True)

#%%
'''-------------------------------------
dark-field
'''
cropDarkfield = darkfield[cropLimitRows[0]:cropLimitRows[1], cropLimitColumns[0]:cropLimitColumns[1]]
climDarkfield = ([0.8, 1.2])
fig1 = figFunc(climDarkfield, cropDarkfield) 
fig2 = plt.hist(cropDarkfield.flatten(), 1000, facecolor='g', alpha=0.75) # alpha range 0-1

#%% 3D image
fig = plt.figure(figsize=plt.figaspect(4.0/8))
ax = fig.add_subplot(111, projection='3d')

xxGrid, yyGrid = np.meshgrid(np.arange(cropDarkfield.shape[0]),
                             np.arange(cropDarkfield.shape[1]),
                             indexing='ij')

surf = ax.plot_surface(xxGrid, yyGrid, cropDarkfield,
                rstride=1, cstride=1,
                cmap='spectral', linewidth=0)
#cmap options: http://mpastell.com/2013/05/02/matplotlib_colormaps/

#plt.xlabel('rows', fontsize=16)
#plt.ylabel('columns', fontsize=16)
ax.set_xlabel('rows')
ax.set_ylabel('columns')
ax.set_zlabel('phi')
#    ax.set_zlim(0, .50)
plt.title('cropDarkfield', fontsize=18, weight='bold')
#plt.colorbar(surf)

plt.show(block=True)

#%%
'''*****************************************************************************
Step 10: Dark-Field: image processing with Inpaint using mask from refVisibilityPercent.  Doesn't help
'''


#%%
'''*****************************************************************************
Step 11: Dark-Field: align sampleVisiblity with referenceVisibility, then calculate Dark-Field
'''


#%%
'''*****************************************************************************
END
'''