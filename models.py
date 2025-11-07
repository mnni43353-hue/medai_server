# models.py
import torch
import torchvision.transforms as T
from torchvision import models
from monai.networks.nets import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vision_model(num_classes=14):
    model = models.resnet50(pretrained=True)
    in_feats = model.fc.in_features
    model.fc = torch.nn.Linear(in_feats, num_classes)
    model.to(DEVICE).eval()
    return model

def load_segmentation_model():
    net = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(DEVICE)
    net.eval()
    return net

def get_image_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])