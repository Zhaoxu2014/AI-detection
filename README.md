# AI-detection

## Papers:


### Paper1:

**Title**: [Extracting Training Data from Diffusion Models](https://www.sciencedirect.com/science/article/abs/pii/S0045790622006851) <br>
**Journal/Conference**: USENIX Security 23 <br>
**Author**: Nicholas Carlini, Google; Jamie Hayes, DeepMind; Milad Nasr and Matthew Jagielski, Google; Vikash Sehwag, Princeton University; Florian Tramèr, ETH Zurich; Borja Balle, DeepMind; Daphne Ippolito, Google; Eric Wallace, UC Berkeley. <br>
**Model**: Diffusion Models <br>
**Dataset**: / <br>
**Method**: / <br>
**Task + evaluation indicators**: 任务是开发和验证一种从扩散模型中提取训练数据的方法。评价指标可能包括提取数据的准确性、完整性和实用性。


### Paper2:

**Title**: [A Reproducible Extraction of Training Images from Diffusion Models](https://arxiv.org/abs/2305.08694) <br>
**Journal/Conference**:  <br>
**Author**: Ryan Webster *Unicaen* <br>
**Model**: Diffusion Models <br>
**Dataset**: LAION-2B, LAION-2B数据集包含了大量的图像-文本对，这些数据对用于训练扩散模型以生成高质量的图像。<br>
**Method**: 论文提出了一种高效的提取攻击方法，可以在较少的网络评估次数下，从扩散模型中提取训练样本。这种方法包括白盒攻击和黑盒攻击，并引入了一个新的现象，称为模板逐字复制（Template Verbatims），即扩散模型在很大程度上完整地复制训练样本。 <br>
**Task + evaluation indicators**: 任务是研究如何从流行的扩散模型中提取训练图像，特别是那些在模型训练集中被复制的图像。评价指标通过构建真实样本的ground truth，并计算模型生成的图像与这些真实样本之间的匹配度（例如，使用均方误差MSE）来评估提取攻击的精度。此外，还通过与先前方法的比较来评估新方法的效率和准确性。


### Paper3:

**Title**: [Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models](https://arxiv.org/abs/2212.03860) <br>
**Journal/Conference**: CVPR 2023 <br>
**Author**: Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, Tom Goldstein. <br>
**Model**: Diffusion Models <br>
**Dataset**: Oxford flowers, Celeb-A, ImageNet, LAION. <br>
**Method**: 论文提出了一种图像检索框架，用于比较生成的图像与训练样本，并检测内容复制的情况。作者考虑了一系列图像相似性度量方法，并使用真实和合成数据集对不同的图像特征提取器进行了基准测试。 <br>
**Task + evaluation indicators**: 任务是检测扩散模型生成的图像是否复制了训练集中的内容，以及复制的程度。评价指标使用mean-Average Precision (mAP)来衡量不同模型在复制检测任务上的性能。此外，通过定性和定量分析来评估模型在不同数据集上生成图像的复制行为。


### Paper4:

**Title**: [Reducing training sample memorization in gans by training with memorization rejection](https://arxiv.org/pdf/2210.12231) <br>
**Journal/Conference**:  <br>
**Author**: Andrew Bai, Cho-Jui Hsieh, Wendy Kan, Hsuan-Tien Lin. <br>
**Model**: GANs <br>
**Dataset**: CIFAR10. CIFAR10是一个广泛用于图像识别和生成模型研究的标准数据集，每个类别有5,000个32x32的彩色图像。<br>
**Method**: 论文提出了一种名为“记忆拒绝”（Memorization Rejection, MR）的训练方案，该方案在训练过程中拒绝那些与训练样本高度相似的生成样本。这种方法简单、通用，并且可以直接应用于任何GAN架构。 <br>
**Task + evaluation indicators**: 任务是研究如何减少GAN在训练过程中对训练样本的记忆现象，以提高生成样本的多样性和质量。评价指标使用Fréchet Inception Distance (FID) 来评估生成质量，使用非参数测试分数（CT值）来评估记忆的严重程度。通过改变拒绝阈值（τ）来平衡生成质量和记忆减少。


### Paper5:

**Title**: [Differentially private diffusion models](https://arxiv.org/pdf/2210.09929) <br>
**Journal/Conference**:  <br>
**Author**: Tim Dockhorn, *Stability AI*, Tianshi Cao, *NVIDIA University of Toronto Vector Institute*, Arash Vahdat, *NVIDIA*, Karsten Kreis, *NVIDIA*. <br>
**Model**: DPDM <br>
**Dataset**: MNIST, Fashion-MNIST, CelebA. <br>
**Method**: 论文提出了一种新的训练方案，即在训练过程中拒绝与训练样本高度相似的生成样本，以此来减少模型对训练数据的记忆现象。引入了一种称为“噪声多样性”的技术，通过对单个训练数据样本在扩散过程中的多个扰动级别进行重用，来提高学习效率，且不会增加额外的隐私成本。 <br>
**Task + evaluation indicators**: 任务是研究如何在保护训练数据隐私的同时，生成高质量的合成数据。评价指标使用Fréchet Inception Distance (FID) 来评估生成样本的质量，以及使用分类器在合成数据上的表现来评估数据的实用性。此外，还使用了非参数统计测试（如Mann-Whitney U测试）来评估模型对训练样本的记忆程度。


### Paper6:

**Title**: [Differentially private diffusion models generate useful synthetic images](https://arxiv.org/pdf/2302.13861) <br>
**Journal/Conference**:  <br>
**Author**: Sahra Ghalebikesabi, Leonard Berrada, Sven Gowal, Ira Ktena, Robert Stanforth, Jamie Hayes, Soham De, Samuel L. Smith, Olivia Wiles, Borja Balle. <br>
**Model**: DPDM <br>
**Dataset**: MNIST, CIFAR-10, Camelyon17. MNIST是一个包含手写数字的图像数据集；CIFAR-10是一个包含10个类别的32x32彩色图像数据集；Camelyon17是一个包含淋巴结组织病理图像的数据集，用于医学图像分析。<br>
**Method**: 论文提出了一种新的训练方案，即在训练过程中拒绝与训练样本高度相似的生成样本，以此来减少模型对训练数据的记忆现象。论文还提出了一种称为“噪声多样性”的技术，通过对单个训练数据样本在扩散过程中的多个扰动级别进行重用，来提高学习效率。 <br>
**Task + evaluation indicators**: 任务是研究如何减少GAN在训练过程中对训练样本的记忆现象，以提高生成样本的多样性和质量。评价指标使用Fréchet Inception Distance (FID) 来评估生成样本的质量和多样性，以及使用分类器在合成数据上的表现来评估数据的实用性。此外，还使用了非参数统计测试（如Mann-Whitney U测试）来评估模型对训练样本的记忆程度。


### Paper7:

**Title**: [DCFace: Synthetic Face Generation with Dual Condition Diffusion Model](https://arxiv.org/pdf/2304.07060) <br>
**Journal/Conference**: CVPR 2023 <br>
**Author**: Minchul Kim, Feng Liu, Anil Jain, Xiaoming Liu. <br>
**Model**: DCFace <br>
**Dataset**: CASIA-WebFace. 数据集包含多个人脸图像，用于训练和评估人脸识别模型。<br>
**Method**: DCFace通过两个阶段的数据生成范式工作：条件采样阶段和混合阶段。在条件采样阶段，使用身份图像生成器Gid生成高质量的身份图像(Xid)，并从风格库中选择一个任意的风格图像(Xsty)。在混合阶段，使用双条件生成器Gmix结合这两种条件生成图像，预测具有Xid身份和Xsty风格的图像。 <br>
**Task + evaluation indicators**: 任务是生成用于训练人脸识别模型的合成数据集，同时确保数据集中的多样性和一致性。评价指标使用人脸识别模型在多个测试数据集上（如LFW, CFP-FP, CPLFW, AgeDB和CALFW）的验证准确率来评估合成图像的性能。


### Paper8:

**Title**: [DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection](https://arxiv.org/pdf/2305.13625) <br>
**Journal/Conference**:  <br>
**Author**: Jiang Liu, Chun Pong Lau, Rama Chellappa. <br>
**Model**: Diffusion Models, Diffusion Autoencoder <br>
**Dataset**: CelebA-HQ, FFHQ. 常用的高质量面部图像数据集。<br>
**Method**: DiffProtect首先将输入面部图像编码为高级语义代码和低级噪声代码。然后，通过迭代优化语义代码来生成能够欺骗面部识别模型的受保护图像。引入了面部语义正则化模块，以鼓励受保护图像和输入图像具有相似的面部语义，以更好地保留视觉身份。提出了一种攻击加速策略，通过仅运行一步生成过程来计算每次攻击迭代的重构图像的近似版本，从而显著减少攻击时间。 <br>
**Task + evaluation indicators**: 任务是在不降低视觉质量的情况下，生成能够欺骗面部识别系统的对抗性面部图像，以保护个人隐私。评价指标使用攻击成功率（ASR）来评估攻击性能，并使用Frechet Inception Distance（FID）来评估受保护面部图像的自然度。


### Paper9:

  **Title**: [A RECIPE FOR WATERMARKING DIFFUSION MODELS](https://arxiv.org/pdf/2303.10137) <br>
**Journal/Conference**:  <br>
**Author**: Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Ngai-Man Cheung, Min Lin. <br>
**Model**: Diffusion Models <br>
**Dataset**: CIFAR-10, FFHQ-70K, AFHQv2, ImageNet-1K. CIFAR-10是一个常用的小型图像数据集，包含10个类别的60000张32x32彩色图像。FFHQ-70K是一个大规模的人脸图像数据集，包含70000多张高分辨率的人脸图像。AFHQv2是一个动物面孔数据集，包含不同种类动物的图像。ImageNet-1K是ImageNet数据集的一个子集，包含1000个类别的图像。<br>
**Method**: 论文提出了一种为扩散模型（DMs）添加水印的方法，以解决版权保护和生成内容监控的法律问题。作者提出了两种水印处理流程：一种是针对无条件/类别条件的DMs，另一种是针对文本到图像的DMs。对于无条件/类别条件的DMs，通过在训练数据中嵌入二进制水印字符串并重新训练模型。对于文本到图像的DMs，通过微调预训练的模型，并使用特定的文本提示和水印图像对来植入水印。 <br>
**Task + evaluation indicators**: 任务是研究如何在扩散模型生成的图像中嵌入水印，以便于版权保护和内容监控。评价指标使用比特准确率（Bit-Acc）来衡量从生成的图像中恢复水印的正确性。此外，还使用了峰值信噪比（PSNR）、结构相似性（SSIM）和Fréchet Inception Distance（FID）来评估生成图像的质量。论文还探讨了水印的鲁棒性，通过在模型权重或生成的图像上添加噪声来测试水印的稳定性。


### Paper10:

**Title**: [Watermarking Diffusion Model](https://arxiv.org/pdf/2305.12502) <br>
**Journal/Conference**:  <br>
**Author**: Yugeng Liu, Zheng Li, Michael Backes, Yun Shen, Yang Zhang. <br>
**Model**: Diffusion Models <br>
**Dataset**: MS COCO (Microsoft Common Objects in Context). 这是一个大规模的对象检测、分割、关键点检测和图像描述数据集。数据集包含118K训练图像和5K验证图像。<br>
**Method**: 论文提出了两种水印方法，NAIVEWM和FIXEDWM，用于保护扩散模型的知识产权。NAIVEWM方法通过在文本提示中注入水印触发词并使用预训练的LDM进行微调来实现水印。FIXEDWM方法更为高级和隐蔽，只有在输入提示中特定位置包含触发词时才能激活水印。 <br>
**Task + evaluation indicators**: 任务是开发一种能够将水印嵌入到扩散模型生成的图像中的方法，以便可以追踪和验证图像的来源。评价指标使用Fréchet Inception Distance (FID)、Structural Similarity Index (SSIM)、Peak Signal-to-Noise Ratio (PSNR)、Visual Information Fidelity (VIFp) 和 Feature-SIMilarity (FSIM) 来评估生成图像的质量。使用均方误差（MSE）来衡量水印图像的质量。


### Paper11:

**Title**: [Securing Deep Generative Models with Universal Adversarial Signature](https://arxiv.org/pdf/2305.16310) <br>
**Journal/Conference**:  <br>
**Author**: Yu Zeng, Mo Zhou, Yuan Xue, Vishal M. Patel. <br>
**Model**: Deep Generative Models, include LDM（Latent Diffusion Models）、ADM（Autoregressive Diffusion Models）and StyleGAN2 <br>
**Dataset**: FFHQ, ImageNet. 前者是一个大规模的人脸图像数据集，包含多种人脸图像；后者是一个广泛使用的图像识别数据集，包含多种类别的图像。<br>
**Method**: 论文提出了一种通过注入通用对抗性签名（Universal Adversarial Signature）来保护深度生成模型的方法。首先，通过对抗训练找到一个对每个图像都不可见的最优签名，然后通过微调任意预训练的生成模型，将签名嵌入到模型中。对应的检测器可以重用于任何微调后的生成器，用于追踪生成器的身份。 <br>
**Task + evaluation indicators**: 任务是开发一种能够将水印嵌入到扩散模型生成的图像中的方法，以便可以追踪和验证图像的来源。评价指标使用峰值信噪比（PSNR）、Fréchet Inception Distance（FID）和分类准确率（Accuracy）来评估生成图像的质量和水印的有效性。此外，还考虑了模型的泛化能力和对图像变换的鲁棒性。


### Paper12:

**Title**: [Cifake: Image classification and explainable identification of ai-generated synthetic images](https://arxiv.org/pdf/2303.14126) <br>
**Journal/Conference**: IEEE <br>
**Author**: Jordan J. Bird, Ahmad Lotfi. <br>
**Model**: LDMs(Latent Diffusion Models), SDM(Stable Diffusion Model). <br>
**Dataset**: 生成了一个新的合成数据集CIFAKE。<br>
**Method**: 研究提出了使用卷积神经网络（CNN）对图像进行分类，将其分为“真实”或“AI生成”两类。研究还包括了超参数调整和36个不同网络拓扑的训练。此外，研究还通过梯度类激活映射（Gradient Class Activation Mapping, Grad-CAM）实施了可解释AI，以探索图像中哪些特征有助于分类。 <br>
**Task + evaluation indicators**: 研究的主要任务是提高我们识别AI生成图像的能力。评价指标包括分类准确率、精确度、召回率和F1分数。通过这些指标，研究评估了CNN在分类真实和AI生成图像方面的性能。


### Paper13:

**Title**: [Improving Synthetically Generated Image Detection in Cross-Concept Settings](https://arxiv.org/pdf/2304.12053) <br>
**Journal/Conference**: MAD ’23 <br>
**Author**: P. Dogoulis, G. Kordopatis-Zilos, I. Kompatsiaris, and S. Papadopoulos. <br>
**Model**: StyleGAN2, Latent Diffusion. <br>
**Dataset**: FFHQ（人脸）、AFHQ（动物）、LSUN（场景和对象类别）。<br>
**Method**: 提出了一种基于图像质量评分的采样策略，用于选择用于训练合成图像检测器的生成图像。使用了一种称为Quality Calculation (QC)的方法来评估生成图像的质量，并根据这些评分来选择训练数据。训练了一个基于ResNet-50的分类器，用于区分真实图像和合成图像。 <br>
**Task + evaluation indicators**: 任务是在跨概念设置中检测合成图像，即训练检测器以识别某一概念类别的合成图像，并测试其在另一概念类别图像上的性能。评价指标主要使用了AUC（Area Under the Curve），这是一种不依赖于特定阈值的评分方法，适合评估检测器的鲁棒性和泛化能力。进行了实验，比较了使用随机采样和基于QC评分的采样策略的训练检测器的性能。














