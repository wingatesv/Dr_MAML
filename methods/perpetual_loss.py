import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50FeatureExtractor, self).__init__()
        # Load the pre-trained ResNet50 model
        resnet50 = models.resnet50(pretrained=pretrained)
        
        # Extract specific layers of ResNet50
        self.layer1 = nn.Sequential(*list(resnet50.children())[:4])  # First few layers
        self.layer2 = nn.Sequential(*list(resnet50.children())[4:5])  # ResNet block 1
        self.layer3 = nn.Sequential(*list(resnet50.children())[5:6])  # ResNet block 2
        self.layer4 = nn.Sequential(*list(resnet50.children())[6:7])  # ResNet block 3
        
        # Set the model to evaluation mode
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Extract features from each layer
        features_layer1 = self.layer1(x)
        features_layer2 = self.layer2(features_layer1)
        features_layer3 = self.layer3(features_layer2)
        features_layer4 = self.layer4(features_layer3)
        
        # Return features from multiple layers
        return [features_layer1, features_layer2, features_layer3, features_layer4]


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Initialize the ResNet50 feature extractor
        self.feature_extractor = ResNet50FeatureExtractor(pretrained=True)
        self.criterion = nn.MSELoss()  # You can switch to nn.L1Loss() if needed

    def forward(self, reconstructed_images, target_images):
        # Extract features from both reconstructed and target images
        reconstructed_features = self.feature_extractor(reconstructed_images)
        target_features = self.feature_extractor(target_images)

        # Compute the perceptual loss as the sum of MSE between feature maps from each layer
        loss = 0
        for recon_feat, target_feat in zip(reconstructed_features, target_features):
            loss += self.criterion(recon_feat, target_feat)
        
        return loss
