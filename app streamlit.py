import streamlit as st
import pandas as pd
import torch
from skimage import io
from utils import plot_closest_imgs
from PIL import Image
from torch import nn
import timm
import numpy as np
import tqdm as tq


class APN_Model(nn.Module):
    def __init__(self,emb_size=512):
        super(APN_Model,self).__init__()
        
        self.efficientnet=timm.create_model('efficientnet_b0',pretrained=True)
        self.efficientnet.classifier=nn.Linear(in_features=self.efficientnet.classifier.in_features,out_features=emb_size)
        
    def forward(self,image):
        embeddings=self.efficientnet(image)
        
        return embeddings

def euclidean_dist(img_enc, anc_enc_arr):
        dist = np.sqrt(np.dot(img_enc - anc_enc_arr, (img_enc - anc_enc_arr).T))
        return dist  

def find_closest_images(input_image, df_enc):
    img = io.imread(input_image)
    img = torch.from_numpy(img).permute(2, 0, 1) / 255.0

    model.eval()
    with torch.no_grad():
        img = img.to(device)  
        img_enc = model(img.unsqueeze(0))  
        img_enc = img_enc.detach().cpu().numpy()

    anc_enc_arr = df_enc.iloc[:, 1:].to_numpy()
    anc_img_names = df_enc['Anchor']

   
    d_dir = 'train/'
    distance = []
    for i in range(anc_enc_arr.shape[0]):
        dist = euclidean_dist(img_enc, anc_enc_arr[i:i + 1, :])
        distance = np.append(distance, dist)

    closest_distance = np.argsort(distance)

    # plot_closest_imgs1(anc_img_names, d_dir, img, input_image, closest_distance, distance, no_of_closest=5)

    images_per_row=5
    image_width=100
    paths=plot_closest_imgs(anc_img_names, d_dir, img, input_image, closest_distance, distance, no_of_closest=5)
    num_rows=len(plot_closest_imgs(anc_img_names, d_dir, img, input_image, closest_distance, distance, no_of_closest=5))//images_per_row
    for row in range(num_rows):
        cols=st.columns(images_per_row*2-1)
        for col_idx,col in enumerate(cols):
            if col_idx%2==0:
                img_path=paths[row*images_per_row+col_idx//2]
                col.image(d_dir+img_path,caption=f"Image {row*images_per_row+col_idx//2+1}", width=image_width,use_column_width=False)
            else:
                col.empty()
    print(paths)
    # for i in plot_closest_imgs(anc_img_names, d_dir, img, input_image, closest_distance, distance, no_of_closest=5):
    #     st.image(d_dir+i, caption='Closest Images', use_column_width=True)
    
def get_encoding_csv(model,anc_img_names):
    anc_img_names_arr=np.array(anc_img_names)
    encodings=[]
    d_dir='train/'
    model.eval()
    with torch.no_grad():
        for i in tq.tqdm(anc_img_names_arr):
            A=io.imread(d_dir+i)
            A=torch.from_numpy(A).permute(2,0,1)/255.0
            A=A.to(device)
            A_enc=model(A.unsqueeze(0)) # adding batchsize
            encodings.append(A_enc.squeeze().cpu().detach().numpy())
        encodings=np.array(encodings)
        encodings=pd.DataFrame(encodings)
        df_enc=pd.concat([anc_img_names,encodings],axis=1)
    
    return df_enc    

def main():
    st.title("Person Re-Identification App")
    st.text("Upload an image to find the closest matches in the database:")
    
    no=st.number_input('Enter image no 0-3999: ', min_value=0,step=1, max_value=3999)
    idx=int(no)
    img_name = df_enc['Anchor'].iloc[idx]
    img_path = 'train/' + img_name

        
    input_image = Image.open(img_path)
    img_path1=input_image.resize((220,250))
    
    

    st.image(img_path1, caption='Uploaded Image.', use_column_width=False,width=450)

    # Save the uploaded image 
    # input_image.save(img_path)

    st.write("Finding closest images...")

    # Display the closest images
    find_closest_images(img_path, df_enc)
if __name__ == '__main__':
    device = torch.device('cuda') 
    df=pd.read_csv('train (1).csv')
    model = APN_Model(1024).to(device)  # Move the model to gpu
    model.load_state_dict(torch.load('best_model.pt'))  # Load the model weights

    df_enc=get_encoding_csv(model,df['Anchor'])  
    df_enc.to_csv('database.csv',index=False)
    # df_enc = pd.read_csv('database (2).csv')

    main()
