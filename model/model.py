import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.nn.parameter import Parameter


class FModel(nn.Module):
    def __init__(self, n_channels=3, bilinear=False, letters_dim = 128, patches_dim =128, num_of_classes = 300):
        super(FModel, self).__init__()
        self.patches_cnn = resnet18(num_classes= patches_dim)
        self.letters_cnn = resnet18(num_classes = letters_dim)
        self.attension_patches = Parameter(torch.ones(patches_dim))
        self.attension_letters = Parameter(torch.ones(letters_dim))
        self.fc = torch.nn.Linear(letters_dim+patches_dim, num_of_classes)
        self.softmax = torch.nn.Softmax()


    def forward(self, patches, letters):
        out_patches = []
        for f in patches:
            out_patches.append(self.patches_cnn(f) )

        out_patches = torch.stack(out_patches)
        final_patches = torch.mean(out_patches, dim=0)
        final_patches =  final_patches * self.attension_patches

        out_letters = []
        for f in letters:
            out_letters.append(self.letters_cnn(f))

        out_letters = torch.stack(out_letters)
        final_letters = torch.mean(out_letters, dim=0)
        final_letters = final_letters * self.attension_letters

        final_vec = torch.cat([final_letters, final_patches], dim=1)
        output = self.fc(final_vec)
        output = self.softmax(output)
        return output


















