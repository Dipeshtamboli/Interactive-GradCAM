# Interactive-GradCAM
The base code for this GradCAM visualization is written by [Jacob Gildenblat](https://github.com/jacobgil). The original code was only for the PyTorch's pretrained Resnet-50 model. I have modified it for the customizable architectures([Modified VGG16 pretrained on COCO dataset](/Interactive_GradCAM_COCO.ipynb)).

### [Colab notebook link for Imagenet](https://colab.research.google.com/github/Dipeshtamboli/Interactive-GradCAM/blob/master/Interactive_GradCAM_ImageNet.ipynb#scrollTo=49m5KXP3r2CC)
### [Colab notebook link for COCO](https://colab.research.google.com/github/Dipeshtamboli/Interactive-GradCAM/blob/master/Interactive_GradCAM_COCO.ipynb)

This repository contains two iPython notebooks.
You can upload your own image and check the GradCAM visualization for the avaialable class.
## Interactive_GradCAM_ImageNet.ipynb:   
   PyTorch's Resnet-50 model trained on Imagenet Dataset 
   Here, you need to give the class number for specifying the target class. Here is the look-up dictionary from which you can check the class index corresponding to its name.
   [Imagenet labels lookup dict](/imagenet_labels.json)  

## Interactive_GradCAM_COCO.ipynb:   
   Modified-VGG16 model(customized) trained on COCO dataset   
   Here, you have to define your architecture and layers where you want to hook for the GradCAM visualization

### Modified VGG16 
```
class VGG16Backbone(nn.Module):
    def __init__(self, num_classes=80):
        super(VGG16Backbone, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.block1 = nn.Sequential(
            *list(vgg.features.children())[:10]
        )
        self.block2 = nn.Sequential(
            *list(vgg.features.children())[10:17]
        )
        self.block3 = nn.Sequential(
            *list(vgg.features.children())[17:24]
        )
        self.block4 = nn.Sequential(
            *list(vgg.features.children())[24:]
        )
        self.conv = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        y = self.conv(x)
        x = F.max_pool2d(y, kernel_size=y.size()[2:])
        last_fc = x.view(x.size(0), -1)
        x = self.fc(last_fc)
        return x, y, last_fc
```

### Model Output
Then change the final flattening layer. In my case, I am doing global maxpool after the `conv` layer. You need to check and change according to you architectur.
```
class ModelOutputs():
    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "conv" in name.lower():
                x = module(x)
                x = F.max_pool2d(x, kernel_size=x.size()[2:])
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        return target_activations, x
```
In the code for Resnet50, flattening layer is after `avgpool`.

```
class ModelOutputs():
    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x
```

@misc{Dipeshtamboli,
  author = {Dipesh Tamboli},
  title = {Interactive GradCAM},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Dipeshtamboli/Interactive-GradCAM}},
  commit = {4834d52554cc3fcbcb8758750957080e250cf4b6}
}
