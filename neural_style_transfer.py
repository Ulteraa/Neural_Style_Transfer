import torch
import torch.nn as nn
from torchvision import  transforms
import torch.optim as optim
from torchvision import  models
from PIL import  Image
from torchvision.utils import save_image
# model=models.vgg19(pretrained=True)
# print(model)
class NST(nn.Module):
    def __init__(self):
        super(NST, self).__init__()
        self.model=models.vgg19(pretrained=True).features[:29]
        self.chosen_feature=['0','5','10','19','28']
    def forward(self,x):
        feature=[]
        # print(self.model)
        for _,layer in enumerate(self.model):
            x=layer(x)
            if str(_) in self.chosen_feature:
                feature.append(x)
        return feature
def load(path):
    img=Image.open(path)
    transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    img=transform(img).unsqueeze(0)

    return img
def train():
    path_original='1.jpg'
    path_style='style/14.jpg'
    epoches=601;lr=0.001
    model=NST()
    origin_img=load(path_original)
    style_img=load(path_style)
    gen_img=origin_img.clone()
    gen_img.requires_grad=True
    optimizer=optim.Adam([gen_img],lr=lr)
    alpha=1
    beta=0.1

    for epoch in range(epoches):
        origin_feature = model(origin_img)
        style_feature = model(style_img)
        gen_feature = model(gen_img)
        content_loss=style_loss=0
        for origin_f, style_f, gen_f in zip(origin_feature,style_feature,gen_feature):
            _, chanel, width, height = origin_f.shape
            content_loss+=torch.mean((origin_f-gen_f)**2)
            style_gram = style_f.view(chanel, width * height).mm(style_f.view(chanel, width * height).t())
            gen_gram = gen_f.view(chanel, width * height).mm(gen_f.view(chanel, width * height).t())

            style_loss+=torch.mean((gen_gram-style_gram)**2)

        total_loss=alpha*content_loss+beta*style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch%10==0:
            print(f'loss is equal to {total_loss}')
            save_image(gen_img,'generated_image.png')
if __name__=='__main__':
    train()







