# AI-detection

## Papers:


### Paper1:

**Title**: [Extracting Training Data from Diffusion Models](https://arxiv.org/pdf/2301.13188) <br>
**Journal/Conference**: USENIX Security 23 <br>
**Author**: Nicholas Carlini, Google; Jamie Hayes, DeepMind; Milad Nasr and Matthew Jagielski, Google; Vikash Sehwag, Princeton University; Florian Tramèr, ETH Zurich; Borja Balle, DeepMind; Daphne Ippolito, Google; Eric Wallace, UC Berkeley. <br>
**Model**: Diffusion Models <br>
**Dataset**: LAION, CIFAR-10. LAION是一个大规模的数据集，用于训练扩散模型；CIFAR-10是一个常用的小规模图像分类数据集，包含10个类别，共60000张32x32的彩色图像。<br>
**Method**: 展示了扩散模型能够记住训练数据中的个别图像，并在生成时重新发出这些图像。提出了一种生成和过滤流程，从最先进的模型中提取了数千个训练样本。训练了数百个不同设置的扩散模型，分析了不同的建模和数据决策如何影响隐私。<br>
**Task + evaluation indicators**: 任务是从预训练的大规模扩散模型中提取训练数据，以评估这些模型对训练数据的记忆程度。评价指标可能包括提取数据的准确性、完整性和实用性。<br>


### Paper2:

**Title**: [A Reproducible Extraction of Training Images from Diffusion Models](https://arxiv.org/abs/2305.08694) <br>
**Journal/Conference**:  <br>
**Author**: Ryan Webster *Unicaen* <br>
**Model**: Diffusion Models <br>
**Dataset**: LAION-2B, LAION-2B数据集包含了大量的图像-文本对，这些数据对用于训练扩散模型以生成高质量的图像。<br>
**Method**: 论文提出了一种高效的提取攻击方法，可以在较少的网络评估次数下，从扩散模型中提取训练样本。这种方法包括白盒攻击和黑盒攻击，并引入了一个新的现象，称为模板逐字复制（Template Verbatims），即扩散模型在很大程度上完整地复制训练样本。 <br>
**Task + evaluation indicators**: 任务是研究如何从流行的扩散模型中提取训练图像，特别是那些在模型训练集中被复制的图像。评价指标通过构建真实样本的ground truth，并计算模型生成的图像与这些真实样本之间的匹配度（例如，使用均方误差MSE）来评估提取攻击的精度。此外，还通过与先前方法的比较来评估新方法的效率和准确性。<br>


### Paper3:

**Title**: [Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models](https://arxiv.org/abs/2212.03860) <br>
**Journal/Conference**: CVPR 2023 <br>
**Author**: Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, Tom Goldstein. <br>
**Model**: Diffusion Models <br>
**Dataset**: Oxford flowers(包含8189张花卉图像的数据集), Celeb-A(包含202599张名人面部图像的数据集), ImageNet(大规模的图像分类数据集，包含超过1400万张标记图像), LAION(一个大规模的、公开可用的图像-文本配对数据集). <br>
**Method**: 论文提出了一种图像检索框架，用于比较生成的图像与训练样本，并检测内容复制的情况。作者考虑了一系列图像相似性度量方法，并使用真实和合成数据集对不同的图像特征提取器进行了基准测试。 <br>
**Task + evaluation indicators**: 任务是检测扩散模型生成的图像是否复制了训练集中的内容，以及复制的程度。评价指标使用mean-Average Precision (mAP)来衡量不同模型在复制检测任务上的性能。此外，通过定性和定量分析来评估模型在不同数据集上生成图像的复制行为。<br>


### Paper4:

**Title**: [Reducing training sample memorization in gans by training with memorization rejection](https://arxiv.org/pdf/2210.12231) <br>
**Journal/Conference**:  <br>
**Author**: Andrew Bai, Cho-Jui Hsieh, Wendy Kan, Hsuan-Tien Lin. <br>
**Model**: GANs <br>
**Dataset**: CIFAR10. CIFAR10是一个广泛用于图像识别和生成模型研究的标准数据集，每个类别有5,000个32x32的彩色图像。<br>
**Method**: 论文提出了一种名为“记忆拒绝”（Memorization Rejection, MR）的训练方案，该方案在训练过程中拒绝那些与训练样本高度相似的生成样本。这种方法简单、通用，并且可以直接应用于任何GAN架构。 <br>
**Task + evaluation indicators**: 任务是研究如何减少GAN在训练过程中对训练样本的记忆现象，以提高生成样本的多样性和质量。评价指标使用Fréchet Inception Distance (FID) 来评估生成质量，使用非参数测试分数（CT值）来评估记忆的严重程度。通过改变拒绝阈值（τ）来平衡生成质量和记忆减少。<br>


### Paper5:

**Title**: [Rdp-gan: A r ́enyi-differential privacy based generative adversarial network](https://arxiv.org/pdf/2007.02056) <br>
**Journal/Conference**: IEEE <br>
**Author**: Chuan Ma, Member, IEEE, Jun Li, Senior Member, IEEE, Ming Ding, Senior Member, IEEE, Bo Liu, Senior Member, IEEE, Kang Wei, Student Member, IEEE, Jian Weng, Member, IEEE, and H. Vincent Poor, Fellow, IEEE. <br>
**Model**: RDP-GAN <br>
**Dataset**: Adult(包含约30,000条个人信息记录，分为14个属性（实验中选择了8个），分为20,000训练样本和10,000测试样本), MNIST(包含70,000个28x28大小的手写数字图像，分为60,000训练样本和10,000测试样本). <br>
**Method**: RDP-GAN通过在训练过程中对判别器的损失函数值添加随机噪声来实现差分隐私，从而在保护训练样本隐私的同时生成高隐私保护的真实样本，提出了自适应噪声调整算法（Adaptive Noise Tuning），根据测试准确性调整添加噪声的量，以改善学习性能。。<br>
**Task + evaluation indicators**: 任务是提出一种新的差分隐私保护的生成对抗网络（RDP-GAN），以解决在敏感或私有训练样本上应用GAN时可能泄露个人隐私信息的问题。评价指标 在MNIST数据集上，使用额外的分类器测试准确性来评估生成样本的质量；在Adult数据集上，进行了概率质量函数（PMF）和绝对平均误差的评估，以及使用训练好的分类器测试准确性；比较了在不同隐私级别（ϵtotal = 0.5 和 ϵtotal = 5）下的算法性能。<br>


### Paper6:

**Title**: [Differentially private diffusion models](https://arxiv.org/pdf/2210.09929) <br>
**Journal/Conference**:  <br>
**Author**: Tim Dockhorn, *Stability AI*, Tianshi Cao, *NVIDIA University of Toronto Vector Institute*, Arash Vahdat, *NVIDIA*, Karsten Kreis, *NVIDIA*. <br>
**Model**: DPDMs(Differentially Private Diffusion Models) <br>
**Dataset**: MNIST(一个包含手写数字的公共数据库，用于训练和测试图像识别算法), Fashion-MNIST(一个替代MNIST的时尚产品图像数据库，同样用于训练和测试图像识别算法), CelebA(一个大型的人脸识别数据集，包含多张名人的人脸图片). <br>
**Method**: 论文提出了一种新的训练方案，即在训练过程中拒绝与训练样本高度相似的生成样本，以此来减少模型对训练数据的记忆现象。引入了一种称为“噪声多样性”的技术，通过对单个训练数据样本在扩散过程中的多个扰动级别进行重用，来提高学习效率，且不会增加额外的隐私成本。 <br>
**Task + evaluation indicators**: 任务是研究如何在保护训练数据隐私的同时，生成高质量的合成数据。评价指标使用Fréchet Inception Distance (FID) 来评估生成样本的质量，以及使用分类器在合成数据上的表现来评估数据的实用性。此外，还使用了非参数统计测试（如Mann-Whitney U测试）来评估模型对训练样本的记忆程度。<br>


### Paper6:

**Title**: [Differentially private diffusion models generate useful synthetic images](https://arxiv.org/pdf/2302.13861) <br>
**Journal/Conference**:  <br>
**Author**: Sahra Ghalebikesabi, Leonard Berrada, Sven Gowal, Ira Ktena, Robert Stanforth, Jamie Hayes, Soham De, Samuel L. Smith, Olivia Wiles, Borja Balle. <br>
**Model**: DPDM <br>
**Dataset**: MNIST, CIFAR-10, Camelyon17. MNIST是一个包含手写数字的图像数据集；CIFAR-10是一个包含10个类别的32x32彩色图像数据集；Camelyon17是一个包含淋巴结组织病理图像的数据集，用于医学图像分析。<br>
**Method**: 论文提出了一种新的训练方案，即在训练过程中拒绝与训练样本高度相似的生成样本，以此来减少模型对训练数据的记忆现象。论文还提出了一种称为“噪声多样性”的技术，通过对单个训练数据样本在扩散过程中的多个扰动级别进行重用，来提高学习效率。 <br>
**Task + evaluation indicators**: 任务是研究如何减少GAN在训练过程中对训练样本的记忆现象，以提高生成样本的多样性和质量。评价指标使用Fréchet Inception Distance (FID) 来评估生成样本的质量和多样性，以及使用分类器在合成数据上的表现来评估数据的实用性。此外，还使用了非参数统计测试（如Mann-Whitney U测试）来评估模型对训练样本的记忆程度。<br>


### Paper7:

**Title**: [Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/papers/Zhang_Forget-Me-Not_Learning_to_Forget_in_Text-to-Image_Diffusion_Models_CVPRW_2024_paper.pdf) <br>
**Journal/Conference**: Gong Zhang, Kai Wang, Xingqian Xu, Zhangyang Wang, Humphrey Shi <br>
**Author**: CVPR 2024 <br>
**Model**: Text-to-Image Diffusion Models <br>
**Dataset**: LAION, COYO, CC12M. <br>
**Method**: 提出 Forget-Me-Not 方法，用于在预训练的文本到图像模型中选择性地忘记特定概念；引入注意力重定向损失（Attention Re-steering loss）和视觉去噪损失（Visual Denoising loss）；使用概念反转（Concept Inversion）技术来提取图像中的文本嵌入。<br>
**Task + evaluation indicators**: 任务是实现在文本到图像的生成模型中有选择性地忘记特定概念，如身份、对象、风格等。评价指标为CLIP得分：评估生成图像与文本提示之间的一致性；记忆得分（Memorization Score）：评估模型对探测数据集的知识变化。 实验包括定性比较、定量分析、概念校正和去除 NSFW 内容的能力评估。对比其他方法如 ESD 和 ACTD，展示了在多概念遗忘和概念校正方面的优势。<br>


### Paper8:

**Title**: [DCFace: Synthetic Face Generation with Dual Condition Diffusion Model](https://arxiv.org/pdf/2304.07060) <br>
**Journal/Conference**: CVPR 2023 <br>
**Author**: Minchul Kim, Feng Liu, Anil Jain, Xiaoming Liu. <br>
**Model**: DCFace(基于Diffusion models) <br>
**Dataset**: CASIA-WebFace. 数据集包含多个人脸图像，用于训练和评估人脸识别模型。<br>
**Method**: DCFace通过两个阶段的数据生成范式工作：条件采样阶段和混合阶段。在条件采样阶段，使用身份图像生成器Gid生成高质量的身份图像(Xid)，并从风格库中选择一个任意的风格图像(Xsty)。在混合阶段，使用双条件生成器Gmix结合这两种条件生成图像，预测具有Xid身份和Xsty风格的图像。 <br>
**Task + evaluation indicators**: 任务是生成用于训练人脸识别模型的合成数据集，同时确保数据集中的多样性和一致性。评价指标使用人脸识别模型在多个测试数据集上（如LFW, CFP-FP, CPLFW, AgeDB和CALFW）的验证准确率来评估合成图像的性能。<br>


### Paper9:

**Title**: [DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection](https://arxiv.org/pdf/2305.13625) <br>
**Journal/Conference**:  <br>
**Author**: Jiang Liu, Chun Pong Lau, Rama Chellappa. <br>
**Model**: DiffProtect(基于Diffusion Models), Diffusion Autoencoder(扩散自编码器) <br>
**Dataset**: CelebA-HQ, FFHQ. 常用的高质量面部图像数据集。<br>
**Method**: DiffProtect首先将输入面部图像编码为高级语义代码和低级噪声代码。然后，通过迭代优化语义代码来生成能够欺骗面部识别模型的受保护图像。引入了面部语义正则化模块，以鼓励受保护图像和输入图像具有相似的面部语义，以更好地保留视觉身份。提出了一种攻击加速策略，通过仅运行一步生成过程来计算每次攻击迭代的重构图像的近似版本，从而显著减少攻击时间。 <br>
**Task + evaluation indicators**: 任务是在不降低视觉质量的情况下，生成能够欺骗面部识别系统的对抗性面部图像，以保护个人隐私。评价指标使用攻击成功率（ASR）来评估攻击性能，并使用Frechet Inception Distance（FID）来评估受保护面部图像的自然度。<br>


### Paper10:

**Title**: [3D-Aware Adversarial Makeup Generation for Facial Privacy Protection](https://arxiv.org/pdf/2306.14640) <br>
**Journal/Conference**:  <br>
**Author**: Yueming Lyu, Yue Jiang, Ziwen He, Bo Peng, Yunfan Liu, Jing Dong <br>
**Model**: 3D-Aware Adversarial Makeup Generation GAN（3DAM-GAN,基于GAN） <br>
**Dataset**: Makeup Transfer (MT) 作为训练集，该数据集包含1,115张无妆容图像和2,719张具有多样妆容风格的图像；测试集包括LADN数据集和CelebA-HQ数据集，用于评估方法的有效性和鲁棒性。<br>
**Method**: 3DAM-GAN通过在UV空间中转移参考面部图像的妆容样式到源图像，从而在保持面部身份的同时，生成具有自然妆容的对抗性面部图像；引入了Makeup Adjustment Module (MAM) 和 Makeup Transfer Module (MTM) 来提高生成图像的质量；提出了一种新的UV妆容损失函数，利用人脸在UV空间的对称性，提供更精确和鲁棒的妆容监督；引入了针对不同面部识别模型的集合训练策略，以提高模型在黑盒设置下的迁移能力。<br>
**Task + evaluation indicators**: 任务是保护面部图像免受未经授权的面部识别系统的识别。评价指标为攻击成功率（Attack Success Rate, ASR）：衡量生成的对抗性图像在面部识别系统中的成功欺骗率；Frechet Inception Distance (FID)：衡量生成图像与真实图像分布之间的相似度；结构相似性指数（Structural Similarity Index Measure, SSIM）和峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）：衡量生成图像的质量。 实验包括与现有方法的比较、不同攻击方法的迁移能力评估、以及在模拟现实场景下的性能测试。<br>


### Paper11:

  **Title**: [A Recipe for Watermarking Diffusion Models](https://arxiv.org/pdf/2303.10137) <br>
**Journal/Conference**:  <br>
**Author**: Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Ngai-Man Cheung, Min Lin. <br>
**Model**: Diffusion Models <br>
**Dataset**: CIFAR-10, FFHQ-70K, AFHQv2, ImageNet-1K. CIFAR-10是一个常用的小型图像数据集，包含10个类别的60000张32x32彩色图像。FFHQ-70K是一个大规模的人脸图像数据集，包含70000多张高分辨率的人脸图像。AFHQv2是一个动物面孔数据集，包含不同种类动物的图像。ImageNet-1K是ImageNet数据集的一个子集，包含1000个类别的图像。<br>
**Method**: 论文提出了一种为扩散模型（DMs）添加水印的方法，以解决版权保护和生成内容监控的法律问题。作者提出了两种水印处理流程：一种是针对无条件/类别条件的DMs，另一种是针对文本到图像的DMs。对于无条件/类别条件的DMs，通过在训练数据中嵌入二进制水印字符串并重新训练模型。对于文本到图像的DMs，通过微调预训练的模型，并使用特定的文本提示和水印图像对来植入水印。 <br>
**Task + evaluation indicators**: 任务是研究如何在扩散模型生成的图像中嵌入水印，以便于版权保护和内容监控。评价指标使用比特准确率（Bit-Acc）来衡量从生成的图像中恢复水印的正确性。此外，还使用了峰值信噪比（PSNR）、结构相似性（SSIM）和Fréchet Inception Distance（FID）来评估生成图像的质量。论文还探讨了水印的鲁棒性，通过在模型权重或生成的图像上添加噪声来测试水印的稳定性。<br>


### Paper12:

**Title**: [The Stable Signature: Rooting Watermarks in Latent Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/papers/Fernandez_The_Stable_Signature_Rooting_Watermarks_in_Latent_Diffusion_Models_ICCV_2023_paper.pdf) <br>
**Journal/Conference**: ICCV <br>
**Author**: Pierre Fernandez，Guillaume Couairon，Hervé Jégou，Matthias Douze，Teddy Furon <br>
**Model**: Diffusion Models, Latent Diffusion Models <br>
**Dataset**: ImageNet(包括了超过100万张高分辨率、多样化的图像)，用于训练潜扩散模型，并用于评估所提出方法的有效性。 <br>
**Method**: 该论文提出了一种名为“Stable Signature”的方法，用于在潜扩散模型生成的图像中嵌入水印；方法涉及在VAE的解码器中嵌入水印，并在训练过程中优化，以便生成的图像包含隐形水印。<br>
**Task + evaluation indicators**: 主要任务是是将水印嵌入到潜扩散模型中，以便在图像生成时能够检测到这些水印，用于版权保护和内容验证。评价指标使用了不可见水印的鲁棒性（Robustness to common image manipulations）；还使用了图像质量评价标准，如Inception Score (IS) 和Fréchet Inception Distance (FID)；通过视觉和自动检测器的检测成功率来评估水印的可检测性。<br>


### Paper13:

**Title**: [Intellectual Property Protection of Diffusion Models via the Watermark Diffusion Process](https://arxiv.org/pdf/2306.03436) <br>
**Journal/Conference**:  <br>
**Author**: Sen Peng, Yufei Chen, Cong Wang, Xiaohua Jia. <br>
**Model**: 文章提出了一种新的水印方案，称为水印扩散过程（Watermark Diffusion Process, WDP），用于保护扩散模型（Diffusion Models）的知识产权。 <br>
**Dataset**: CIFAR-10(常用的图像识别数据集，包含10个类别，每个类别6000张32x32的彩色图像), CelebA(大规模的人脸属性数据集，包含超过20万张人脸图像及其40种属性的注释). <br>
**Method**: 该方法通过在训练过程中同时学习水印扩散过程（WDP）和标准扩散过程来嵌入水印数据；WDP允许模型在不损害原始任务生成质量的前提下，生成具有独特数据分布的样本作为水印；提供了WDP训练和抽样的详细理论分析，并与修改的高斯扩散过程通过相同的反向噪声联系起来。<br>
**Task + evaluation indicators**: 任务是开发一种完整的水印框架，用于保护扩散模型的知识产权，包括水印嵌入、提取和验证。评价指标分为模型保真度：通过Inception Score (IS)和Fréchet Inception Distance (FID)评估生成模型的质量，以确保水印过程没有显著影响模型性能；水印保真度：通过比较提取的水印和原始水印的相似性，以及使用假设检验验证水印的存在；水印鲁棒性：评估水印在面对模型压缩、权重扰动和模型微调等攻击时的鲁棒性。<br>


### Paper14:

**Title**: [Watermarking Diffusion Model](https://arxiv.org/pdf/2305.12502) <br>
**Journal/Conference**:  <br>
**Author**: Yugeng Liu, Zheng Li, Michael Backes, Yun Shen, Yang Zhang. <br>
**Model**: Diffusion Models <br>
**Dataset**: MS COCO (Microsoft Common Objects in Context). 这是一个大规模的对象检测、分割、关键点检测和图像描述数据集。数据集包含118K训练图像和5K验证图像。<br>
**Method**: 论文提出了两种水印方法，NAIVEWM和FIXEDWM，用于保护扩散模型的知识产权。NAIVEWM方法通过在文本提示中注入水印触发词并使用预训练的LDM进行微调来实现水印。FIXEDWM方法更为高级和隐蔽，只有在输入提示中特定位置包含触发词时才能激活水印。 <br>
**Task + evaluation indicators**: 任务是开发一种能够将水印嵌入到扩散模型生成的图像中的方法，以便可以追踪和验证图像的来源。评价指标使用Fréchet Inception Distance (FID)、Structural Similarity Index (SSIM)、Peak Signal-to-Noise Ratio (PSNR)、Visual Information Fidelity (VIFp) 和 Feature-SIMilarity (FSIM) 来评估生成图像的质量。使用均方误差（MSE）来衡量水印图像的质量。<br>


### Paper15:

**Title**: [Securing Deep Generative Models with Universal Adversarial Signature](https://arxiv.org/pdf/2305.16310) <br>
**Journal/Conference**:  <br>
**Author**: Yu Zeng, Mo Zhou, Yuan Xue, Vishal M. Patel. <br>
**Model**: Deep Generative Models, include LDM（Latent Diffusion Models）、ADM（Autoregressive Diffusion Models）and StyleGAN2 <br>
**Dataset**: FFHQ, ImageNet. 前者是一个大规模的人脸图像数据集，包含多种人脸图像；后者是一个广泛使用的图像识别数据集，包含多种类别的图像。<br>
**Method**: 论文提出了一种通过注入通用对抗性签名（Universal Adversarial Signature）来保护深度生成模型的方法。首先，通过对抗训练找到一个对每个图像都不可见的最优签名，然后通过微调任意预训练的生成模型，将签名嵌入到模型中。对应的检测器可以重用于任何微调后的生成器，用于追踪生成器的身份。 <br>
**Task + evaluation indicators**: 任务是开发一种能够将水印嵌入到扩散模型生成的图像中的方法，以便可以追踪和验证图像的来源。评价指标使用峰值信噪比（PSNR）、Fréchet Inception Distance（FID）和分类准确率（Accuracy）来评估生成图像的质量和水印的有效性。此外，还考虑了模型的泛化能力和对图像变换的鲁棒性。<br>


### Paper16:

**Title**: [DiffusionShield: A Watermark for Data Copyright Protection against Generative Diffusion Models](https://arxiv.org/pdf/2306.04642) <br>
**Journal/Conference**:  <br>
**Author**: Yingqian Cui, Jie Ren, Han Xu, Pengfei He, Hui Liu, Lichao Sun, Yue Xing, Jilijang Tang. <br>
**Model**: 文章提出了一种名为DiffusionShield的新方法，用于保护图像版权免受生成扩散模型（Generative Diffusion Models, GDMs）的侵权。 <br>
**Dataset**: CIFAR-10, CIFAR-100, STL-10, ImageNet-20. <br>
**Method**: DiffusionShield方法包括两个阶段：保护阶段和审计阶段。在保护阶段，版权所有者将版权信息编码成水印并添加到图像中，形成受保护的数据；在审计阶段，版权所有者检查可疑图像是否侵犯了其数据的版权DiffusionShield通过块状策略增强水印的“模式一致性”，并通过联合优化方法提高水印检测性能。<br>
**Task + evaluation indicators**: 任务是开发和评估一种用于保护图像版权免受生成扩散模型侵权的水印方法。评价指标分别为扰动预算（Perturbation Budget）：使用LPIPS、l2和l∞差异来衡量原始和水印图像之间的视觉差异；检测准确率（Detection Accuracy）：应用位准确率来评估编码在生成图像中的版权信息的正确性；消息长度（Message Length）：反映编码能力的容量，即嵌入水印中的消息长度。<br>


### Paper17:

**Title**: [Cifake: Image classification and explainable identification of ai-generated synthetic images](https://arxiv.org/pdf/2303.14126) <br>
**Journal/Conference**: IEEE <br>
**Author**: Jordan J. Bird, Ahmad Lotfi. <br>
**Model**: LDMs(Latent Diffusion Models), SDM(Stable Diffusion Model). <br>
**Dataset**: 生成了一个新的合成数据集CIFAKE。<br>
**Method**: 研究提出了使用卷积神经网络（CNN）对图像进行分类，将其分为“真实”或“AI生成”两类。研究还包括了超参数调整和36个不同网络拓扑的训练。此外，研究还通过梯度类激活映射（Gradient Class Activation Mapping, Grad-CAM）实施了可解释AI，以探索图像中哪些特征有助于分类。 <br>
**Task + evaluation indicators**: 研究的主要任务是提高我们识别AI生成图像的能力。评价指标包括分类准确率、精确度、召回率和F1分数。通过这些指标，研究评估了CNN在分类真实和AI生成图像方面的性能。<br>


### Paper18:

**Title**: [On the Detection of Synthetic Images Generated by Diffusion Models](https://arxiv.org/pdf/2211.00680) <br>
**Journal/Conference**: ICASSP <br>
**Author**: R Corvi, D Cozzolino, G Zingarini, G Poggi, K Nagano, L Verdoliva <br>
**Model**: 论文主要研究的是扩散模型（Diffusion Models，DM）生成的合成图像的检测问题，而非提出一个新的生成模型。 <br>
**Dataset**: 使用了多个最新的生成模型生成的合成图像，包括但不限于ProGAN、StyleGAN2、StyleGAN3、BigGAN、EG3D、Taming Transformer、DALL·E Mini、DALL·E 2、GLIDE、Latent Diffusion、Stable Diffusion和ADM。真实图像数据集包括COCO、ImageNet和UCID。<br>
**Method**: 论文首先分析了扩散模型留下的法医痕迹，然后研究了当前针对GAN生成图像开发的检测器在这些新型合成图像上的表现，尤其是在社交网络环境中涉及图像压缩和调整大小的挑战性场景。<br>
**Task + evaluation indicators**: 任务：检测由扩散模型生成的合成图像。评价指标：使用接收者操作特征曲线下面积（AUC）和在固定阈值为0.5时的准确度来评估检测器的性能。<br>


### Paper19:

**Title**: [DIRE for Diffusion-Generated Image Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_DIRE_for_Diffusion-Generated_Image_Detection_ICCV_2023_paper.pdf) <br>
**Journal/Conference**: ICCV 2023 <br>
**Author**: Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, Houqiang Li. <br>
**Model**: 论文提出了一种新的图像表示方法，称为DIffusion Reconstruction Error (DIRE)，用于检测由扩散模型 (Diffusion Models) 生成的图像。 <br>
**Dataset**: 作者建立了一个名为DiffusionForensics的数据集，用于评估检测扩散生成图像的性能，数据集包括由各种扩散模型生成的图像，包括无条件、条件和文本到图像的扩散生成模型。数据集涵盖了LSUN-Bedroom、ImageNet和CelebA-HQ三个域的图像。<br>
**Method**: DIRE方法基于预训练的扩散模型对输入图像进行重建，并计算输入图像与其重建对应物之间的误差。扩散生成的图像可以近似地由扩散模型重建，而真实图像则不能。通过训练一个简单的二元分类器基于DIRE可以轻松检测扩散生成的图像。<br>
**Task + evaluation indicators**: 任务是开发一种通用的检测器，用于区分真实图像和由扩散模型生成的图像。评价指标：使用准确率（Accuracy）和平均精度（Average Precision，AP）来评估检测器的性能。 实验包括在DiffusionForensics数据集上的广泛实验，以证明DIRE表示在检测扩散生成图像方面的优越性。<br>


### Paper20:

**Title**: [Improving Synthetically Generated Image Detection in Cross-Concept Settings](https://arxiv.org/pdf/2304.12053) <br>
**Journal/Conference**: 2nd ACM International Workshop on Multimedia AI against Disinformation (MAD ’23) <br>
**Author**: P. Dogoulis, G. Kordopatis-Zilos, I. Kompatsiaris, and S. Papadopoulos. <br>
**Model**: StyleGAN2, Latent Diffusion. <br>
**Dataset**: FFHQ（人脸）、AFHQ（动物）、LSUN（场景和对象类别）。<br>
**Method**: 提出了一种基于图像质量评分的采样策略，用于选择用于训练合成图像检测器的生成图像。使用了一种称为Quality Calculation (QC)的方法来评估生成图像的质量，并根据这些评分来选择训练数据。训练了一个基于ResNet-50的分类器，用于区分真实图像和合成图像。 <br>
**Task + evaluation indicators**: 任务是在跨概念设置中检测合成图像，即训练检测器以识别某一概念类别的合成图像，并测试其在另一概念类别图像上的性能。评价指标主要使用了AUC（Area Under the Curve），这是一种不依赖于特定阈值的评分方法，适合评估检测器的鲁棒性和泛化能力。进行了实验，比较了使用随机采样和基于QC评分的采样策略的训练检测器的性能。<br>


### Paper21:

**Title**: [Detecting Images Generated by Deep Diffusion Models using their Local Intrinsic Dimensionality](https://openaccess.thecvf.com/content/ICCV2023W/DFAD/papers/Lorenz_Detecting_Images_Generated_by_Deep_Diffusion_Models_Using_Their_Local_ICCVW_2023_paper.pdf) <br>
**Journal/Conference**: ICCV 2023 <br>
**Author**: Peter Lorenz, Ricard L. Durall, Janis Keuper, Janis Keuper. <br>
**Model**: 该论文并未提出新的生成模型，而是提出了一种检测方法，用于检测和识别由深度扩散模型（如DDPMs）生成的图像。 <br>
**Dataset**: 使用了多个数据集，包括公开可用的CiFake、ArtiFact、DiffusionDB、LAION-5B和SAC数据集，以及作者从不同预训练模型生成的新数据集。数据集包含多种图像尺寸（从32×32到768×768像素）和不同领域的图像，如面部、动物、地点等。 <br>
**Method**: 提出了一种基于局部内在维度（multiLID）的方法，用于自动检测合成图像并识别相应的生成网络。方法包括将输入图像传递给未训练的ResNet，提取特征映射表示，计算multiLID，然后运行分类器以确定输入图像的性质。<br>
**Task + evaluation indicators**: 任务为检测合成图像与真实图像的区分以及识别合成图像是由哪个特定的生成模型生成的。评价指标：使用准确率（Accuracy）作为评价指标，并通过一系列实验验证了所提方法的有效性，包括在标准化数据集（如LSUN-Bedroom）和最新数据集上的性能评估，以及对multiLID方法的彻底研究。<br>


### Paper22:

**Title**: [Level up the deepfake detection: a method to effectively discriminate images generated by gan architectures and diffusion models](https://arxiv.org/pdf/2303.00608) <br>
**Journal/Conference**: Intelligent Systems Conference, 2024 <br>
**Author**: L Guarnera, O Giudice, S Battiato. <br>
**Model**: 该论文不是关于生成模型，而是提出了一种用于区分由不同生成对抗网络（GAN）架构和扩散模型（DM）生成的图像的检测方法。 <br>
**Dataset**: 数据集包括从CelebA、FFHQ和ImageNet数据集收集的真实图像，以及由9种不同的GAN引擎（如AttGAN、CycleGAN、StyleGAN等）和4种文本到图像的DM架构（如DALL-E 2、Latent Diffusion等）生成的合成数据。总共收集了42,500张合成图像和40,500张真实图像。<br>
**Method**: 提出了一种多级层次方法，利用ResNet模型来解决三个不同的深度伪造检测和识别任务：真实与AI生成图像的区分（Level 1）；GANs与DMs的区分（Level 2）；特定AI（GAN/DM）架构的识别（Level 3）。<br>
**Task + evaluation indicators**: 任务是区分真实图像与AI生成图像、区分由不同GAN和DM生成的图像，以及识别特定AI架构。评价指标是分类准确率，实验结果显示在每个任务中都超过了97%的准确率，超越了现有的最先进方法。实验中使用了ResNet-18和ResNet-34模型，并比较了两种模型的性能。<br>


### Paper23:

**Title**: [Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models](https://arxiv.org/pdf/2305.13873) <br>
**Journal/Conference**: ACM Conference on Computer and Communications Security (CCS) <br>
**Author**: Yiting Qu, Xinyue Shen, Xinlei He, Michael Backes, Yang Zhang, Savvas Zannettou . <br>
**Model**: 论文中研究了多种文本到图像模型，包括Stable Diffusion、DALL·E 2、Latent Diffusion、DALL·E mini等，以及它们在生成不安全图像和仇恨模因方面的风险。 <br>
**Dataset**: 作者构建了一个包含五种类别（性暗示、暴力、令人不安、仇恨、政治）的不安全图像类型学。使用了来自4chan、Lexica网站以及手动创建的基于模板的数据集，这些数据集可能产生有害提示。还包括了MSCOCO和Flickr30k数据集，用于生成和评估图像。 <br>
**Method**: 研究了文本到图像模型在给定提示下生成不安全内容（包括仇恨模因）的风险；提出了一种用于检测不安全内容的方法，并对Stable Diffusion模型进行了评估，以生成特定个体或社区的仇恨模因变体；使用了三种图像编辑方法：DreamBooth、Textual Inversion和SDEdit，这些都是由Stable Diffusion支持的。<br>
**Task + evaluation indicators**: 任务1（RQ1）：安全评估，检测文本到图像模型在敌手滥用情况下生成不安全内容的风险。任务2（RQ2）：仇恨模因生成，评估敌手能否利用文本到图像模型生成仇恨模因。评价指标包括准确率、查准率、查全率、F1分数等，以及构建了一个多头图像安全分类器来检测定义范围内的不安全图像。<br>


### Paper24:

**Title**: [Mitigating Inappropriateness in Image Generation: Can there be Value in Reflecting the World’s Ugliness?](https://arxiv.org/pdf/2305.18398) <br>
**Journal/Conference**:  <br>
**Author**: Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting. <br>
**Model**: 文中评估了多种文本到图像的生成模型，包括不同版本的Stable Diffusion、AltDiffusion、MultiFusion、Paella和Deepfloyd-IF等，这些模型都使用了*扩散模型 (Diffusion Models)* 技术。 <br>
**Dataset**: 作者使用了不适当图像提示数据集 (I2P) 进行实验，该数据集包含4700多个真实用户提示，这些提示很可能生成不适当的图像。I2P数据集涵盖了仇恨、骚扰、暴力、自残、性内容、震惊等类别。 <br>
**Method**: 文章提出了在推理时评估缓解策略以抑制不适当内容生成的方法。特别是，提出了两种指导方法：负面提示 (negative prompting) 和语义引导 (SEGA)，以指导模型在图像生成过程中避免生成不适当的内容。<br>
**Task + evaluation indicators**: 任务是评估和缓解文本到图像生成模型可能产生的不适当内容。评价指标包括生成不适当内容的概率和预期最大不适当性。使用了Q16分类器和NudeNet来自动评估图像内容的不适当性。实验结果表明，通过直接指示可以有效地抑制不适当内容的生成，并且SEGA在减少不适当内容生成方面比负面提示更有效。 <br>


### Paper25:

**Title**: [Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models](https://openaccess.thecvf.com/content/CVPR2023/papers/Schramowski_Safe_Latent_Diffusion_Mitigating_Inappropriate_Degeneration_in_Diffusion_Models_CVPR_2023_paper.pdf) <br>
**Journal/Conference**: CVPR 2023 <br>
**Author**: Patrick Schramowski, Manuel Brack, Björn Deiseroth, Kristian Kersting. <br>
**Model**: 论文提出的是一种名为Safe Latent Diffusion (SLD) 的方法，用于减轻*扩散模型 (如Stable Diffusion)* 在图像生成过程中可能出现的不适当内容。 <br>
**Dataset**: Inappropriate Image Prompts (I2P) ，包含4703个真实世界的文本到图像提示，这些提示可能会生成不适当的图像内容，如裸露和暴力。<br>
**Method**: SLD 方法在不经过额外训练的情况下，利用分类器自由引导（classifier-free guidance）的原理，在图像生成过程中移除或抑制不适当的图像部分；SLD 通过在生成过程中引入安全引导项（safety guidance term），将生成的图像推向与文本提示一致的方向，同时避免生成定义的不适当概念。。<br>
**Task + evaluation indicators**: 任务是评估和减轻文本到图像生成模型在生成不适当内容方面的风险。评价指标包括生成不适当内容的概率和预期最大不适当性。使用Q16分类器和NudeNet来自动评估图像内容的不适当性。实验结果显示，SLD方法在减少生成不适当内容方面非常有效，与现有的方法相比，在多个测试设置中准确率更高。<br>


### Paper26:

**Title**: [Erasing Concepts from Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/papers/Gandikota_Erasing_Concepts_from_Diffusion_Models_ICCV_2023_paper.pdf) <br>
**Journal/Conference**: ICCV 2023 <br>
**Author**: Rohit Gandikota, Joanna Materzyńska, Jaden Fiotto-Kaufman, David Bau. <br>
**Model**: 该论文提出了一种名为Erased Stable Diffusion (ESD) 的方法，用于从预训练的*扩散模型 (Stable Diffusion)* 中擦除特定概念。这些概念可能包括不希望生成的内容，如性内容、版权艺术风格等。 <br>
**Dataset**: 使用Inappropriate Image Prompts (I2P) 基准测试数据集来评估方法的有效性。I2P 包含4703个可能导致不适当图像生成的文本提示。<br>
**Method**: 提出了一种微调方法，可以在没有额外数据的情况下，仅通过风格名称擦除预训练扩散模型中的视觉概念；使用负面引导作为教师，通过微调模型权重来实现擦除概念；提出了两种配置：ESD-x（擦除特定于文本提示的概念）和ESD-u（擦除与文本提示无关的全局概念）。<br>
**Task + evaluation indicators**: 任务1：评估微调方法在移除不适当内容（如性显式内容）方面的有效性。任务2：评估微调方法在移除特定艺术风格方面的有效性。评价指标包括不适当图像分类的准确性、精确度、召回率和F1分数，以及用户研究来评估艺术风格移除的人类感知。<br>


### Paper27:

**Title**: [Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/376276a95781fa17c177b1ccdd0a03ac-Paper-Conference.pdf) <br>
**Journal/Conference**: NeurIPS 2023 <br>
**Author**: Alvin Heng, Harold Soh. <br>
**Model**: 条件变分似然模型，包括*变分自编码器（VAEs）* 和*去噪扩散概率模型（DDPMs）*。 <br>
**Dataset**: MNIST(手写数字数据集，由0到9的数字灰度图像组成), CIFAR10和STL10(常用的图像分类数据集，包含多种类别的彩色图像). <br>
**Method**: 提出了一种基于持续学习的方法，通过结合弹性权重固化（Elastic Weight Consolidation, EWC）和生成重放（Generative Replay, GR），来训练模型忘记特定概念。该方法允许用户指定忘记的概念，并将其映射到用户定义的更适当的概念。<br>
**Task + evaluation indicators**: 任务是训练模型忘记特定的类别或概念，例如MNIST中的数字0、CIFAR10和STL10中的飞机类别，以及文本到图像模型中的名人和裸露提示。评价指标：图像质量指标：使用Fréchet Inception Distance (FID)、Precision和Recall来评估记忆类别的图像质量。被忘记类别的概率：使用外部分类器评估生成样本中被忘记类别的概率。分类器熵：计算分类器输出分布的平均熵，以评估生成样本中被忘记类别的信息是否被抹去。 实验在不同的模型和数据集上进行实验，包括MNIST、CIFAR10、STL10以及开源的文本到图像模型Stable Diffusion；通过定量和定性分析来评估模型在忘记特定概念的同时，对其他类别图像质量的影响。 <br>


### Paper28:

**Title**: [Red-Teaming the Stable Diffusion Safety Filter](https://arxiv.org/pdf/2210.04610) <br>
**Journal/Conference**: NeurIPS 2022 <br>
**Author**: Javier Rando, Daniel Paleka, David Lindner, Lennard Heim, Florian Tramèr. <br>
**Model**: Stable Diffusion <br>
**Dataset**: 没有明确提到特定的数据集，而是针对Stable Diffusion模型内置的安全过滤器进行了研究和测试。 <br>
**Method**: 逆向工程了*Stable Diffusion*的安全过滤器，揭示了过滤器的设计和工作原理。But 过滤器主要针对的是性内容，而忽略了暴力、血腥和其他同样令人不安的内容。<br>
**Task + evaluation indicators**: 任务：评估Stable Diffusion模型的安全过滤器是否能有效防止生成显式图像。 评价指标：通过实验生成各种可能被过滤器拦截或漏过的图像；使用了一种名为“prompt dilution”（提示稀释）的策略来绕过过滤器；逆向工程了过滤器的嵌入向量，以找出过滤器试图阻止的具体概念；提供了一个公共的Colab笔记本，允许用户测试任何给定图像是否被过滤器分类为不安全。 论文中提到了多个实验，包括生成显式内容（包括暴力、性内容等），并评估这些内容是否被Stable Diffusion的安全过滤器拦截；测试了过滤器对于不同类型的显式内容的反应，发现许多情况下生成的内容能够绕过过滤器。 <br>


### Paper29:

**Title**:  <br>
**Journal/Conference**:  <br>
**Author**:  <br>
**Model**:  <br>
**Dataset**: <br>
**Method**: 。<br>
**Task + evaluation indicators**: <br>


### Paper30:

**Title**:  <br>
**Journal/Conference**:  <br>
**Author**:  <br>
**Model**:  <br>
**Dataset**: <br>
**Method**: 。<br>
**Task + evaluation indicators**: <br>











