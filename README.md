# AI-detection

## Papers:

### Paper1:

**Title**: [A full data augmentation pipeline for small object detection based on generative adversarial networks](https://www.sciencedirect.com/science/article/pii/S0031320322004782#sec0011) PR <br>
**Author**: Brais Bosquet, Daniel Cores, Lorenzo Seidenari, Víctor M. Brea, Manuel Mucientes, Alberto Del Bimbo <br>
**Model**: GAN <br>
**Dataset**: [UAVDT](https://datasetninja.com/uavdt) Selected from 10 hours raw videos, about 80, 000 representative frames are fully annotated with bounding boxes as well as up to 14 kinds of attributes (e.g., weather condition, flying altitude, camera view, vehicle category, and occlusion) for three fundamental computer vision tasks: object detection, single object tracking, and multiple object tracking. <br>
**Method**: <br>
**Task+evaluation indicators**: 

### Paper2:

**Title**: [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html) CVPR <br>
**Author**: Utkarsh Ojha, Yuheng Li, Yong Jae Lee, 威斯康星大学麦迪逊分校. <br>
**Model**: 文中并没有提出一个新的生成模型，而是针对现有的生成模型（如GAN、扩散模型和自回归模型）的输出进行检测。 <br>
**Dataset**: 文中使用了多种生成模型生成的数据集进行实验，包括ProGAN、StyleGAN、BigGAN、CycleGAN、StarGAN、GauGAN、CRN、IMLE、SAN、SITD、DeepFakes、LDM、Glide和DALL-E等。此外，还使用了LAION数据集作为真实图像的来源。 <br>
**Method**: 作者首先分析了现有基于深度学习的方法在检测未见过的生成模型产生的假图像时的局限性。然后，提出了一种不依赖于显式训练来区分真实和假图像的特征空间的方法。具体来说，使用了预训练的视觉-语言模型（如CLIP-ViT）的特征空间，通过最近邻分类和线性探测来进行真实与假图像的分类。 <br>
**Task + evaluation indicators**: 任务是开发一种通用的假图像检测方法，能够检测出任意图像是否为假图像。评价指标包括平均精度（Average Precision, AP）和分类准确率（Accuracy）。通过在多种未见过的生成模型上测试，评估了所提出方法的泛化能力。

### Paper3:
