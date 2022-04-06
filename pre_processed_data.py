
import numpy as np
import cv2 as cv
import os
import pickle   

path,dir,file1=next(os.walk(r"C:\Users\giris\OneDrive\Desktop\project\New folder\2011_09_26\2011_09_26_drive_0001_sync\image_00\data"))
'Input of first five images are eliminated in all sequences'
image1=np.zeros((len(file1)+1,192,640,3));
for i in range(5,len(file1)-4):
     base=r'C:\Users\giris\OneDrive\Desktop\project\New folder\2011_09_26\2011_09_26_drive_0001_sync\image_00\data'
     p1=os.path.join(base,file1[i])
     img1 = cv.imread(p1) 
     img1=cv.resize(img1,(640,192))
     image1[i-5,:,:,:]=img1;
image1=image1[5:len(file1)-5,]

paths,dir,file2=next(os.walk(r"C:\Users\giris\Videos\Project\Train\2011_09_26_drive_0009_sync\image_00\data"))    
image2=np.zeros((len(file2)+1,192,640,3));
for i in range(5,439):
     base=r'C:\Users\giris\Videos\Project\Train\2011_09_26_drive_0009_sync\image_00\data'
     p2=os.path.join(base,file2[i])
     img2 = cv.imread(p2) 
     img2=cv.resize(img2,(640,192))
     image2[i-5,:,:,:]=img2;
image2=image2[5:439,]

 
x=int(len(file1)+len(file2)-20)
img=np.zeros((x,375,1242,3))
img=np.concatenate((image1,image2),axis=0)
'https://www.datacamp.com/community/tutorials/pickle-python-tutorial'
with open("t1.p","wb") as f:
    pickle.dump(img,f)
    f.close()
x=len(file1)+len(file2)-20
img=np.zeros((x,375,1242,3))
img=np.concatenate((image1,image2),axis=0)
with open("v.p","wb") as f:
    pickle.dump(img,f)
    f.close()

img.shape

path,dirss,file3=next(os.walk(r"C:\Users\giris\Videos\Depth\train\2011_09_26_drive_0001_sync\proj_depth\groundtruth\image_02"))   
    
image3=np.zeros((len(file3),192,640,3));
for i in range(0,len(file3)):
    base=r'C:\Users\giris\Videos\Depth\train\2011_09_26_drive_0001_sync\proj_depth\velodyne_raw\image_02'
    p3=os.path.join(base,file3[i])
    img3 = cv.imread(p3) 
    img3=cv.resize(img3,(640,192))
    image3[i,:,:,:]=img3;


path,dirss,file4=next(os.walk(r"C:\Users\giris\Videos\Project\Data\train\2011_09_26_drive_0009_sync\proj_depth\groundtruth\image_02"))    
image4=np.zeros((len(file4)+192,640,3));
for i in range(0,len(file4)):
     base=r'C:\Users\giris\Videos\Project\Data\train\2011_09_26_drive_0009_sync\proj_depth\groundtruth\image_02'
     p4=os.path.join(base,file4[i])
     img4 = cv.imread(p4) 
     img4=cv.resize(img4,(640,192))

     image4[i,:,:,:]=img4;
 
    
x=len(file3)+len(file4)
img=np.zeros((x,375,1242,3))
img=np.concatenate((image3,image4),axis=0)
'https://www.datacamp.com/community/tutorials/pickle-python-tutorial'  
with open("g1.p","wb") as f:
    pickle.dump(img,f)
    f.close()

img.shape

path,dirss,file5=next(os.walk(r"C:\Users\giris\Videos\Project\Train\2011_09_26_drive_0011_sync\image_00\data"))    
image5=np.zeros((len(file5),192,640,3));
for i in range(5,len(file5)-4):
      base=r'C:\Users\giris\Videos\Project\Train\2011_09_26_drive_0011_sync\image_00\data'
      p5=os.path.join(base,file5[i])
      img5 = cv.imread(p5) 
      img5=cv.resize(img5,(640,192))

      image5[i-5,:,:,:]=img5;
image5=image5[5:len(file5)-5,]
 
'https://www.datacamp.com/community/tutorials/pickle-python-tutorial'
with open("t25.p","wb") as f:
    pickle.dump(image5,f)
    f.close()

path,dirss,file6=next(os.walk(r"C:\Users\giris\Videos\Project\Data\train\2011_09_26_drive_0011_sync\proj_depth\groundtruth\image_02"))   
image6=np.zeros((len(file6),375,1242,3));
for i in range(0,len(file6)):
      base=r'C:\Users\giris\Videos\Project\Data\train\2011_09_26_drive_0011_sync\proj_depth\groundtruth\image_02'
      p6=os.path.join(base,file6[i])
      img6 = cv.imread(p6) 
      image6[i,:,:,:]=img6;
'https://www.datacamp.com/community/tutorials/pickle-python-tutorial'      
with open("g25.p","wb") as f:
    pickle.dump(image6,f)
    f.close()

'similarly all the files are pre-processed and stored as pickle files'