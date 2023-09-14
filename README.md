# Depth-Estimation

Depth estimation is the process of determining the distance of objects in an image or video from the camera or observer. In computer vision, monocular depth estimation is a challenging problem as it requires to estimate the depth from single image rather than multiple images from different perspectives. However this can be achieved by using machine learning techniques such as convolutional neural networks(CNNs). In this specific case, the script uses TensorFlow to pre-process the KITTI dataset for monocular depth estimation by applying a series of transformations to the images and labels, such as loading images and depth maps, resizing, normalizing and data augmentation before using a CNN to predict depth maps for new images.

![DepthImage](https://github.com/Girish16/Depth-Estimation/blob/main/depth%20image.jpg?raw=true)
