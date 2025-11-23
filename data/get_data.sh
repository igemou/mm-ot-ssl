mkdir -p ~/scratch/coco && cd ~/scratch/coco

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip train2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip

# Data will take around 20GBs disk space
# train2017 and val2027 just have images in them
# annotations folder has:
#     ├── captions_train2017.json    
#     ├── captions_val2017.json      
#     ├── instances_train2017.json   
#     ├── instances_val2017.json
#     ├── person_keypoints_train2017.json
#     └── person_keypoints_val2017.json
# We only need captions for training and val. 
# The other are used for other COCO tasks like object detection and human pose estimation
