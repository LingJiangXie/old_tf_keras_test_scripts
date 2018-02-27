import os
import shutil
import random
from PIL import Image
import cv2


#targetdir = os.listdir('/home/dany/workspace/mini-webface')
#targettrain = os.listdir('/home/dany/Desktop/train')
#targettest = os.listdir('/home/dany/Desktop/test')



'''
for allDir in targetdir:
        child = os.path.join('%s%s' % ('/home/dany/Desktop/lfw-deepfunneled/', allDir))
        childx= os.listdir(child)

        shutil.move(os.path.join('%s%s%s' % ('/home/dany/Desktop/lfw-deepfunneled/', allDir+'/',childx[-1])), os.path.join('%s%s' % ('/home/dany/Desktop/train/', childx[-1])))
        


for allDir in targetdir:
        child = os.path.join('%s%s' % ('/home/dany/Desktop/lfw-deepfunneled/', allDir))
        childx= os.listdir(child)
        for image in childx:
            #print(os.path.join('%s%s%s' % ('/home/dany/Desktop/lfw-deepfunneled/', allDir+'/',image)))
            #print(os.path.join('%s%s' % ('/home/dany/Desktop/test/', allDir + image)))
            shutil.move(os.path.join('%s%s%s' % ('/home/dany/Desktop/lfw-deepfunneled/', allDir + '/', image)),
                        os.path.join('%s%s' % ('/home/dany/Desktop/test/', image)))


count=0

while(count<4837):
    index=random.randint(0,7483)
    if os.path.exists(os.path.join('%s%s' % ('/home/dany/Desktop/test/',  targettest[index]))):
        shutil.move(os.path.join('%s%s' % ('/home/dany/Desktop/test/',  targettest[index])),os.path.join('%s%s' % ('/home/dany/Desktop/train/', targettest[index])))
        count += 1
'''
   


#newtrain=os.listdir('/home/dany/Desktop/lfw-train')
#newtest=os.listdir('/home/dany/Desktop/lfw-test')


'''
for train_image in targettrain:
    index = train_image.index('0') - 1

    #print(train_image[0:index])
    #print(os.path.join('%s%s' % ('/home/dany/Desktop/train/', train_image)))
    #print(os.path.join('%s%s' % ('/home/dany/Desktop/lfw-train/', train_image[0:index])))

    shutil.move(os.path.join('%s%s' % ('/home/dany/Desktop/train/', train_image)),
                os.path.join('%s%s' % ('/home/dany/Desktop/lfw-train/', train_image[0:index])))



for test_image in targettest:
    index = test_image.index('0') - 1
    print(test_image)
    shutil.move(os.path.join('%s%s' % ('/home/dany/Desktop/test/', test_image)),
                os.path.join('%s%s' % ('/home/dany/Desktop/lfw-test/', test_image[0:index])))



#灰度图
targetdir = os.listdir('/home/dany/Desktop/CASIA-WebFace')
greytargetdir = os.listdir('/home/dany/Desktop/CASIA-grey')

for allDir in targetdir:
        child = os.path.join('%s%s' % ('/home/dany/Desktop/CASIA-WebFace/', allDir))
        childx= os.listdir(child)
        for image in childx:
            #print(os.path.join('%s%s%s' % ('/home/dany/Desktop/lfw-deepfunneled/', allDir+'/',image)))
            #print(os.path.join('%s%s' % ('/home/dany/Desktop/test/', allDir + image)))
            im = Image.open(os.path.join('%s%s%s' % ('/home/dany/Desktop/CASIA-WebFace/', allDir + '/', image))).convert('L')
            
            im.save(os.path.join('%s%s%s' % ('/home/dany/Desktop/CASIA-grey/', allDir + '/', image)))
            






# 灰度图
targetdir = os.listdir('/home/dany/Desktop/CASIA-WebFace')
greytargetdir = os.listdir('/home/dany/Desktop/CASIA-grey')

for allDir in targetdir:
    child = os.path.join('%s%s' % ('/home/dany/Desktop/CASIA-WebFace/', allDir))
    childx = os.listdir(child)
    for image in childx:
        # print(os.path.join('%s%s%s' % ('/home/dany/Desktop/lfw-deepfunneled/', allDir+'/',image)))
        # print(os.path.join('%s%s' % ('/home/dany/Desktop/test/', allDir + image)))
        im = Image.open(os.path.join('%s%s%s' % ('/home/dany/Desktop/CASIA-WebFace/', allDir + '/', image))).convert(
            'L')

        im.save(os.path.join('%s%s%s' % ('/home/dany/Desktop/CASIA-grey/', allDir + '/', image)))



#为每张图添加一个标签
count=0

while(count<504):
    index=random.randint(0,1503)
    if os.path.exists(os.path.join('%s%s' % ('/home/dany/workspace/mini-webface/',  targetdir[index]))):
        childx = os.listdir(os.path.join('%s%s' % ('/home/dany/workspace/mini-webface/',  targetdir[index])))
        shutil.rmtree(os.path.join('%s%s' % ('/home/dany/workspace/mini-webface/',  targetdir[index])))
        count += 1

'''

count=0

while(count<4060):
    index=random.randint(101,6573530)
    if index<1000:

        fatherdir='/home/dany/Desktop/CASIA-WebFace_grey/'+'0000'+str(index)+'/'
        print(fatherdir)
        count += 1
        '''
        if os.path.exists(os.path.join('%s%s' % ('/home/dany/Desktop/CASIA-WebFace_grey/', targettest[index]))):
            shutil.move(os.path.join('%s%s' % ('/home/dany/Desktop/test/', targettest[index])),
                        os.path.join('%s%s' % ('/home/dany/Desktop/train/', targettest[index])))
            count += 1
            
        '''
    else:
        if index<10000:
            fatherdir = '/home/dany/Desktop/CASIA-WebFace_grey/' + '000' + str(index) + '/'
            print(fatherdir)
            count += 1

        else:
            if index<100000:
                fatherdir = '/home/dany/Desktop/CASIA-WebFace_grey/' + '00' + str(index) + '/'
                print(fatherdir)
                count += 1
            else:
                if index < 1000000:
                    fatherdir = '/home/dany/Desktop/CASIA-WebFace_grey/' + '0' + str(index) + '/'
                    print(fatherdir)
                    count += 1
                else:
                    fatherdir = '/home/dany/Desktop/CASIA-WebFace_grey/' + str(index) + '/'
                    print(fatherdir)
                    count += 1

                    








