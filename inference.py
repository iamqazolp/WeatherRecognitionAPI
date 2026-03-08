import torchvision
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
model_path = "weather_resnet_model.pth"

# Verify the model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Please complete the training first.")
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# We don't need to download the default weights anymore so weights=None
model = torchvision.models.resnet18(weights=None) 
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

#load the precious trained-on-cpu model
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #more pain
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

inference_transform = torchvision.transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
from PIL import Image,ImageShow
def path_to__pil_img(image_path):
    pil_image=Image.open(image_path).convert('RGB')
    return pil_image
    

def predict(image):
    # the original image is of the format (h,w,c) but ToTensor automatically handles it
    input_tensor = inference_transform(image)
    #the model expects batch so we use unsqueeze to add a single batch dimension
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output=model(input_batch)
        probs=F.softmax(output[0],dim=0)
        #return the class with the highest probablity
        return torch.max(probs),class_names[torch.argmax(probs).item()]
        #if dumb enough: return class_names[torch.multinomial(probs,1)] to sample
