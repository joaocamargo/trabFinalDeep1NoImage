# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models

filename = "mobilenet_v2_10e_1_16.txt"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device,file=open(filename, "a"))

"""# Importar dataset"""

pathXRayImages =  './chest_xray'
from PIL import Image
std = []
PATHTrain = pathXRayImages + '/train/'
PATHTest = pathXRayImages + '/test/'
PATHVal = pathXRayImages + '/val/'


def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1,2,0)
  image = image * np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
  image = image.clip(0,1)
  return image

def getitem(self, item): 
  image, label = self.imgs[item] 
  image = Image.open(image) 
  img = transform(image) 
  return img, label

classes = (    'NORMAL','PNEUMONIA'    )
	
print('Resize 256',file=open(filename, "a"))
print('randomcrop 224',file=open(filename, "a"))
print('batchsize - 100',file=open(filename, "a"))

transform_train = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.RandomAffine(0,shear=10,scale=(0.8,1.6)),
                                      transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])


transform = transforms.Compose([transforms.Resize((256,256)),
                                transforms.ToTensor(),
                                 transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

training_dataset = datasets.ImageFolder(root=PATHTrain,transform=transform_train)
validation_dataset = datasets.ImageFolder(root=PATHVal,transform=transform)
training_loader = torch.utils.data.DataLoader(dataset=training_dataset,batch_size=100,shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=100,shuffle=False)


test_dataset = datasets.ImageFolder(root=PATHTest,transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=100,shuffle=True)


print(len(training_loader),file=open(filename, "a"))
print(len(validation_loader),file=open(filename, "a"))
print(len(test_loader),file=open(filename, "a"))

dataiter = iter(training_loader)
images,labels = dataiter.next()

#MODEL#MODEL#MODEL#MODEL#MODEL#MODEL#MODEL#MODEL

model = models.mobilenet_v2(pretrained=True)
model

for param in model.features.parameters():
  param.requires_grad =False
  
#mudando apenas para duas classes
n_inputs = model.classifier[1].in_features
last_layer = nn.Linear(n_inputs,len(classes))
model.classifier[1] = last_layer
model.to(device)
print(model)

import time
start_time = time.time()


print('weights = torch.tensor([1.0, 16.0]).to(device)',file=open(filename, "a"))

weights = torch.tensor([1.0, 16.0]).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adagrad(model.parameters(), lr = 0.001)
print('optimizer = torch.optim.Adagrad(model.parameters(), lr = 0.001)',file=open(filename, "a"))


print('levou {} segundos '.format(time.time() - start_time),file=open(filename, "a"))

import time
start_time = time.time()


##EPOCHS###########

print('epochs =10',file=open(filename, "a"))

epochs =10

running_loss_history=[]
running_corrects_history=[]

val_running_loss_history=[]
val_running_corrects_history=[]


for e in range(epochs):
  
  #if e == 6:
   # print('trocando lr de 0,01 para 0,001')
   # optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    
  running_loss = 0.0
  running_corrects = 0.0
  val_running_loss = 0.0
  val_running_corrects = 0.0
  
  for inputs, labels in training_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    
    if len(labels) != len(inputs) or len(inputs) != len(outputs):
      print(len(labels))
      print(len(inputs))
      print(len(outputs))
    
    loss = criterion(outputs,labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    _,preds = torch.max(outputs,1)    
    running_loss += loss.item()
    running_corrects+= torch.sum(preds == labels.data)    
    
  else:
    with torch.no_grad():
      for val_inputs, val_labels in validation_loader:
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs,val_labels)        
        
        _,val_preds = torch.max(val_outputs,1)
        val_running_loss += val_loss.item()
        val_running_corrects+= torch.sum(val_preds == val_labels.data)  
      
    epoch_loss = running_loss/len(training_loader.dataset)
    #epoch_loss = calcula o loss function atual
    epoch_acc = running_corrects.float()/len(training_loader.dataset)
    #epoch_acc = pega a quantiade de acertos que teve em relacao ao total e seta a porcentagem de acertos
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc)
    
    val_epoch_loss = val_running_loss/len(validation_loader.dataset)
    #epoch_loss = calcula o loss function atual
    val_epoch_acc = val_running_corrects.float()/len(validation_loader.dataset)
    #epoch_acc = pega a quantiade de acertos que teve em relacao ao total e seta a porcentagem de acertos
    val_running_loss_history.append(val_epoch_loss)    
    val_running_corrects_history.append(val_epoch_acc)    
    
    print('epoch: ',str(e+1),file=open(filename, "a"))
    print('train_loss: {:.4f},{:.4f}'.format(epoch_loss,epoch_acc.item()),file=open(filename, "a"))
    print('valid_loss: {:.4f}, \nvalid_acc {:.4f}'.format(val_epoch_loss,val_epoch_acc.item()),file=open(filename, "a"))
    #print('difference between loss:', val_epoch_loss-epoch_loss)

    
print('levou {} segundos '.format(time.time() - start_time),file=open(filename, "a"))

dataiter = iter(test_loader)
images,labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)
model.eval()
output = model(images)
_,preds = torch.max(output,1)

#fig = plt.figure(figsize=(20,20))

acertos = 0
erros = 0

for idx in np.arange(len(images)):
  #ax = fig.add_subplot(20,5,idx+1,xticks=[],yticks=[])
  #plt.imshow(im_convert(images[idx]))  
  if preds[idx]==labels[idx]:
    acertos = acertos +1
  else:
    erros = erros +1
  #ax.set_title("{} ({})".format(str(classes[preds[idx].item()]),str(classes[labels[idx].item()])),color=('green' if preds[idx]==labels[idx] else "red"))
  
print('acertos: {} \nerros:{}'.format(acertos,erros),file=open(filename, "a"))

from sklearn.metrics import confusion_matrix
nb_classes = 2

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(test_loader):#['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix,file=open(filename, "a"))

print(confusion_matrix.diag()/confusion_matrix.sum(1),file=open(filename, "a"))

#import seaborn as sn
#import matplotlib.pyplot as plt

#plt.figure(figsize = (10,7))
#sn.heatmap(confusion_matrix, annot=True,fmt='g')

















