# AI-detection

## Papers:

### Paper1:

**Title**: [A full data augmentation pipeline for small object detection based on generative adversarial networks](https://www.sciencedirect.com/science/article/pii/S0031320322004782#sec0011) <br>
**Journal/Conference**: PR <br>
**Author**: Brais Bosquet, Daniel Cores, Lorenzo Seidenari, Víctor M. Brea, Manuel Mucientes, Alberto Del Bimbo <br>
**Model**: GAN <br>
**Dataset**: [UAVDT](https://datasetninja.com/uavdt) Selected from 10 hours raw videos, about 80, 000 representative frames are fully annotated with bounding boxes as well as up to 14 kinds of attributes (e.g., weather condition, flying altitude, camera view, vehicle category, and occlusion) for three fundamental computer vision tasks: object detection, single object tracking, and multiple object tracking. <br>
**Method**: <br>
**Task+evaluation indicators**: 


### Paper2:

**Title**: [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html) <br>
**Journal/Conference**: CVPR <br>
**Author**: Utkarsh Ojha, Yuheng Li, Yong Jae Lee, *威斯康星大学麦迪逊分校* . <br>
**Model**: 文中并没有提出一个新的生成模型，而是针对现有的生成模型（如GAN、扩散模型和自回归模型）的输出进行检测。 <br>
**Dataset**: 文中使用了多种生成模型生成的数据集进行实验，包括ProGAN、StyleGAN、BigGAN、CycleGAN、StarGAN、GauGAN、CRN、IMLE、SAN、SITD、DeepFakes、LDM、Glide和DALL-E等。此外，还使用了LAION数据集作为真实图像的来源。 <br>
**Method**: 作者首先分析了现有基于深度学习的方法在检测未见过的生成模型产生的假图像时的局限性。然后，提出了一种不依赖于显式训练来区分真实和假图像的特征空间的方法。具体来说，使用了预训练的视觉-语言模型（如CLIP-ViT）的特征空间，通过最近邻分类和线性探测来进行真实与假图像的分类。 <br>
**Task + evaluation indicators**: 任务是开发一种通用的假图像检测方法，能够检测出任意图像是否为假图像。评价指标包括平均精度（Average Precision, AP）和分类准确率（Accuracy）。通过在多种未见过的生成模型上测试，评估了所提出方法的泛化能力。


### Paper3:

**Title**: [Extracting Training Data from Diffusion Models](https://www.sciencedirect.com/science/article/abs/pii/S0045790622006851) <br>
**Journal/Conference**: USENIX Security 23 <br>
**Author**: Nicholas Carlini, Google; Jamie Hayes, DeepMind; Milad Nasr and Matthew Jagielski, Google; Vikash Sehwag, Princeton University; Florian Tramèr, ETH Zurich; Borja Balle, DeepMind; Daphne Ippolito, Google; Eric Wallace, UC Berkeley. <br>
**Model**: Diffusion Models <br>
**Dataset**: / <br>
**Method**: / <br>
**Task + evaluation indicators**: 任务是开发和验证一种从扩散模型中提取训练数据的方法。评价指标可能包括提取数据的准确性、完整性和实用性。


### Paper4:

**Title**: [A Reproducible Extraction of Training Images from Diffusion Models](https://www.sciencedirect.com/science/article/abs/pii/S0045790622006851) <br>
**Journal/Conference**: CVPR <br>
**Author**: Ryan Webster *Unicaen* <br>
**Model**: Diffusion Models <br>
**Dataset**: LAION-2B, LAION-2B数据集包含了大量的图像-文本对，这些数据对用于训练扩散模型以生成高质量的图像。<br>
**Method**: 论文提出了一种高效的提取攻击方法，可以在较少的网络评估次数下，从扩散模型中提取训练样本。这种方法包括白盒攻击和黑盒攻击，并引入了一个新的现象，称为模板逐字复制（Template Verbatims），即扩散模型在很大程度上完整地复制训练样本。 <br>
**Task + evaluation indicators**: 任务是研究如何从流行的扩散模型中提取训练图像，特别是那些在模型训练集中被复制的图像。评价指标通过构建真实样本的ground truth，并计算模型生成的图像与这些真实样本之间的匹配度（例如，使用均方误差MSE）来评估提取攻击的精度。此外，还通过与先前方法的比较来评估新方法的效率和准确性。


### Paper5:

**Title**: [Extracting Training Data from Diffusion Models](https://www.sciencedirect.com/science/article/abs/pii/S0045790622006851) <br>
**Journal/Conference**: USENIX Security 23 <br>
**Author**: Nicholas Carlini, Google; Jamie Hayes, DeepMind; Milad Nasr and Matthew Jagielski, Google; Vikash Sehwag, Princeton University; Florian Tramèr, ETH Zurich; Borja Balle, DeepMind; Daphne Ippolito, Google; Eric Wallace, UC Berkeley. <br>
**Model**: Diffusion Models <br>
**Dataset**: / <br>
**Method**: / <br>
**Task + evaluation indicators**: 任务是开发和验证一种从扩散模型中提取训练数据的方法。评价指标可能包括提取数据的准确性、完整性和实用性。












