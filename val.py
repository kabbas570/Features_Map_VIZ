   #### Specify all the paths here #####
   
test_imgs=r'C:\My_Data\trasnformer_code\VIZ\img'
test_masks=r'C:\My_Data\trasnformer_code\VIZ\img'

path_to_checkpoints=R"C:\My_Data\trasnformer_code\VIZ\MY_10_5.pth.tar"

Save_Visual_results= True   ## set to False, if only the quanitative resutls are required
path_to_save_visual_results=r'C:\My_Data\trasnformer_code\VIZ'


        #### Specify all the Hyperparameters\image dimenssions here #####

batch_size=1
height=224
width=224

        #### Import All libraies used for training  #####
import torch    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
import cv2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision

  ### Data_Generators ########
  
NUM_WORKERS=0
PIN_MEMORY=False

class Dataset(Dataset):
    def __init__(self, image_dir, mask_dir,transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        image=cv2.imread(img_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        
        mean=np.mean(image)
        std=np.std(image)
        image=(image-mean)/std
        
        #image=np.transpose(image, (2, 0, 1))
        
        mask=cv2.imread(mask_path,0)
        mask[np.where(mask!=0)]=1
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)
        mask=np.expand_dims(mask, axis=0)

        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        return image,mask,self.images[index][:-4]
    
def Data_Loader( image_dir,mask_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset( image_dir=image_dir, mask_dir=mask_dir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader
    
    
from model10_5 import MY_1
model_2=MY_1()


   ### Load the Data using Data generators and paths specified #####
   #######################################

test_loader=Data_Loader(test_imgs,test_masks,batch_size)
print(len(test_loader)) ### this shoud be = Total_images/ batch size

   ### Evaluation metrics #######

def Evaluation_Metrics(pre,gt):
    pre=pre.flatten() 
    gt=gt.flatten()  
    tn, fp, fn, tp=confusion_matrix(gt,pre,labels=[0,1]).ravel()
    
    iou=tp/(tp+fn+fp) 
    dice=2*tp/(2*tp + fp + fn)
    return iou,dice,tp,tn,fp,fn 
def Evaluate_model(test_loader, model, device=DEVICE):
    dice_score1=0
    dice_score2=0
    Intersection_Over_Union=0
    TPS=0
    TNS=0
    FPS=0
    FNS=0
    
    loop = tqdm(test_loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data,t1,label) in enumerate(loop):
            data = data.to(device=DEVICE,dtype=torch.float)
            t1 = t1.to(device=DEVICE,dtype=torch.float)
            
            p1,Before,After=model(data)
            
            for i in range(10):
                torchvision.utils.save_image(Before[0,i,:,:],str(i)+'Before.png')
                torchvision.utils.save_image(After[0,i,:,:],str(i)+'After.png')
            
            
            print(Before.shape)
            print(After.shape)

            p1 = (p1 > 0.5).float()
            dice_score1 += (2 * (p1 * t1).sum()) / (
                (p1 + t1).sum() + 1e-8
            )

            p1=p1.cpu()
            t1=t1.cpu()
            iou_,dice_,tp,tn,fp,fn =Evaluation_Metrics(p1,t1)
            
            TPS=TPS+tp
            TNS=TNS+tn
            FPS=FPS+fp
            FNS=FNS+fn       
            
            Intersection_Over_Union=Intersection_Over_Union+iou_
            dice_score2=dice_score2+dice_

    print(f"Dice score for Segmentation of Steel from Formula:1 : {dice_score1/len(test_loader)}")
    print(f"Dice score for Segmentation of Steel from Formula:2 : {dice_score2/len(test_loader)}")
    print(f"IoU for Segmentation of Steel: {Intersection_Over_Union/len(test_loader)}")
    print('TPs :',TPS)
    print('TNs :',TNS)
    print('FPs :',FPS)
    print('FNs :',FNS)

### saving the visual results  #####

def blend(image1,gt,pre, ratio=0.5):
    
    assert 0 < ratio <= 1, "'cut' must be in 0 to 1"

    alpha = ratio
    beta = 1 - alpha
    theta=beta-0.1

    #####  coloring yellow the True Positives  #####
    
    gt *= [0.2,0.7, 0] ### Green Color for ground-truth
    pre*=[1,0,0]   ## Red Color for predictions
    
    image = image1 * alpha + gt * beta+ pre * theta
    return image

def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
def save_predictions_as_imgs(
    loader, model, device=DEVICE):
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1,gt1,label) in enumerate(loop):
            img1 = img1.to(device=DEVICE,dtype=torch.float)
            gt1 = gt1.to(device=DEVICE,dtype=torch.float)
            
            p1= model(img1)              
            p1 = (p1 > 0.5).float()   
             
            for k in range(img1.shape[0]):
                img=img1[k,:,:,:]
                t1=gt1[k,0,:,:]
                pre1=p1[k,0,:,:]
                name=label[k]
                                
                img=img.cpu().numpy()
                t1=t1.cpu().numpy()
                pre1=pre1.cpu().numpy()
                                
                img=normalize(img)
                stacked_gt = np.stack((t1,)*3, axis=-1)
                stacked_pre = np.stack((pre1,)*3, axis=-1)
                
                #img=np.transpose(img, (1,2,0))

                result=blend(img,stacked_gt,stacked_pre)

                plt.imsave(os.path.join(path_to_save_visual_results,name+".png"),result)
def eval_():
    model_2.to(device=DEVICE,dtype=torch.float)
    optimizer = optim.Adam(model_2.parameters(), betas=(0.9, 0.999),lr=0)
    checkpoint = torch.load(path_to_checkpoints,map_location=DEVICE)
    model_2.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    Evaluate_model(test_loader, model_2, device=DEVICE)

    if Save_Visual_results:
        save_predictions_as_imgs(test_loader, model_2, device=DEVICE)
    
if __name__ == "__main__":
    eval_()