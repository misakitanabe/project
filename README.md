# 1. Clone repo
git clone https://github.com/misakitanabe/rotten-fruit-detector.git

# 2. Install requirements
pip install -r requirements.txt
(i hope this works sorry if it doesn't!)

# 3. Build your model!
python3.9 <full path to main.py>

  ex) python3.9 /Users/misakitanabe/Documents/Cal\ Poly/year4/CSC\ 466/project/scripts/main.py

Note: Right now, it only uses 3 epochs because it takes too long for more but ultimately, we'd 
      probably want to use more. Also, only transfer learning is implemented, not finetuning. 

This script builds off of the base model, Xception, and uses transfer learning to train the model
to predict good/rotten fruits. Transfer learning freezes the original layers of the pretrained 
model, then adds custom layers for classification. This makes it so that our model will only predict
for 12 classes rather than 1000 like in the original pretrained model.

The 12 classes consist of 6 fruits, good and bad. 

# Using pretrained MobileNetV2 to predict class of good banana image (optional)
This pretrained model is pretrained on ImageNet images so it can classify for these 1000 labels:
https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/

Banana, is one of the classes that it is trained on already, meaning we can predict our good banana
images using this model just to try it out.

To try out the pretrained model as it is, navigate to 'models/base_models/mobile_net_predict_single.py',
and it should output the top 5 predictions for the image. You can try other images of good bananas too,
by changing the path of img in line 19! I think they predict it as a slug right now haha.

# Using pretrained Xception to predict class of good banana image (optional)
This pretrained model is also pretrained on ImageNet images so it can classify for these 1000 labels:
https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/

This model is a lot larger than MobileNetV2, and provided better overall accuracy when predicting 
good bananas therefore I moved forward with this one to use for our model.

To try out the pretrained model as it is, navigate to 'models/base_models/xception_predict_single.py',
and it should output the top 5 predictions for the image. You can try other images of good bananas too,
by changing the path of img in line 19! 
