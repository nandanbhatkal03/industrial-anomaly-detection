# Literature Review

> Source: Phase-I specification, sections 3, 11 and 12, **with corrections** listed in `DOCUMENT_CORRECTIONS.md`.
> References marked ⚠ were not independently verified (see the verification table at the end).


## 3.1 Evolution of Industrial Visual Inspection

Traditional industrial inspection systems generally rely on manually
designed image-processing operations such as thresholding, edge
detection, morphological processing, texture descriptors and template
matching. These methods can work well in controlled environments but may
become difficult to maintain when illumination, product appearance,
surface texture or defect characteristics vary.

Deep learning has substantially changed industrial inspection by
enabling learned feature representations. CNN-based methods have been
used for classification, detection and segmentation, while
anomaly-detection methods increasingly formulate inspection as a
one-class or unsupervised problem.

A 2024 IEEE Access review of CNN-based industrial defect detection
reports the effectiveness of CNNs across industrial applications while
identifying limited datasets, computational complexity and
domain-specific variations as continuing challenges.

**Reference:**\
[A Comprehensive Review of Convolutional Neural Networks for Defect
Detection in Industrial Applications --- IEEE Access,
2024](https://ieeexplore.ieee.org/document/10589380/)

------------------------------------------------------------------------

## 3.2 Industrial Anomaly Detection and the MVTec AD Benchmark

Bergmann et al. introduced **MVTec AD**, a comprehensive real-world
dataset specifically designed for unsupervised industrial anomaly
detection. It contains high-resolution images covering object and
texture categories, with defect-free training images and anomalous test
images. Pixel-accurate anomaly masks make it possible to evaluate both
image-level detection and pixel-level localization.

This work established an important benchmark for subsequent industrial
anomaly-detection research and supports the project proposed here
because the training setting reflects a practical situation in which
normal samples are much easier to obtain than all possible defect types.

**References:** the extended journal version is: Bergmann et al., "The MVTec Anomaly Detection Dataset…", IJCV 129(4):1038–1059, 2021, DOI 10.1007/s11263-020-01400-4 — cite it alongside the CVPR paper (DOI 10.1109/CVPR.2019.00982).\
[Bergmann et al., "MVTec AD --- A Comprehensive Real-World Dataset for
Unsupervised Anomaly Detection," CVPR
2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html)

------------------------------------------------------------------------

## 3.3 Surface Defect Detection Using Deep Learning

Deep-learning-based surface-defect detection has been studied using
supervised, unsupervised and hybrid approaches. An IEEE conference
review categorized deep-learning surface-defect methods and discussed
their advantages, limitations, industrial applications and commonly used
datasets.

The review demonstrates that a major design choice is whether the system
should rely on large numbers of labelled defect samples or learn the
normal appearance of products and detect previously unseen deviations.

**Reference:**\
["Surface Defect Detection Methods Based on Deep Learning: a Brief
Review," IEEE ITCA 2020](https://ieeexplore.ieee.org/document/9421960/)

------------------------------------------------------------------------

## 3.4 PaDiM and Distribution-Based Feature Modeling

PaDiM models the distribution of pretrained CNN features at different
spatial locations. Normal feature statistics are estimated from
defect-free images, and Mahalanobis distance can then be used to
identify anomalous regions.

The approach is important for this project because it demonstrates that
industrial anomaly detection can be performed without requiring a large
collection of labelled defect classes.

**Reference:**\
[Defard et al., "PaDiM: A Patch Distribution Modeling Framework for
Anomaly Detection and Localization," ICPR
2021](https://arxiv.org/abs/2011.08785)

------------------------------------------------------------------------

## 3.5 PatchCore and Memory-Based Anomaly Detection

PatchCore introduced a memory-bank-based approach that stores
representative normal patch features. It uses locally aware (neighbourhood-aggregated) mid-level features and coreset subsampling to reduce redundancy and
computational cost.

PatchCore demonstrated very strong image-level and localization
performance on MVTec AD and became an important baseline for industrial
anomaly detection.

Its memory-bank design is particularly relevant to the proposed project
because **memory usage and inference efficiency** are important
considerations when attempting to create a lightweight system.

**Reference:**\
[Roth et al., "Towards Total Recall in Industrial Anomaly Detection,"
CVPR
2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)

------------------------------------------------------------------------

## 3.6 Self-Supervised Anomaly Detection and Localization

Self-Supervised Masking proposed a reconstruction-oriented approach in
which random masking is used during training and progressive mask
refinement is used during inference. The method addresses both anomaly
detection and localization.

This work demonstrates that self-supervised learning can improve the
usefulness of reconstruction-based methods for localized defects and
reduce reliance on manually labelled anomaly examples.

**Reference:**\
["Self-Supervised Masking for Unsupervised Anomaly Detection and
Localization," IEEE Transactions on Multimedia,
2023](https://ieeexplore.ieee.org/document/9779083/)

------------------------------------------------------------------------

## 3.7 Unsupervised Industrial Anomaly Detection with Self-Updated Memory

An IEEE Transactions on Instrumentation and Measurement paper proposed a
self-updated memory and center-clustering framework for industrial image
anomaly detection and localization. The approach uses pretrained
features, clustering and an adaptive memory mechanism to better
represent the current dataset distribution.

This work highlights an important limitation of directly reusing
pretrained feature representations: the feature distribution of
industrial products can differ from the natural-image data used during
pretraining.

**Reference:**\
["Unsupervised Image Anomaly Detection and Localization in Industry
Based on Self-Updated Memory and Center Clustering," IEEE Transactions
on Instrumentation and Measurement,
2023](https://ieeexplore.ieee.org/document/10114470/)

------------------------------------------------------------------------

## 3.8 EfficientAD and Computationally Efficient Anomaly Detection

EfficientAD directly addresses the computational-efficiency problem. It
introduces a lightweight feature extractor and a student-teacher
anomaly-detection mechanism, together with an efficient autoencoder for
logical anomalies.

The paper reports very low inference latency while maintaining strong
anomaly detection and localization performance across industrial anomaly
datasets.

This is one of the most directly relevant works for the proposed project
because the proposed project explicitly targets the
**accuracy--efficiency trade-off**.

**Reference:**\
[Batzner, Heckler and König, "EfficientAD: Accurate Visual Anomaly
Detection at Millisecond-Level Latencies," WACV
2024](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html)

------------------------------------------------------------------------

## 3.9 Lightweight Industrial Anomaly Detection

Recent IEEE research has also explored lightweight anomaly-detection
architectures. For example, a 2025 IEEE Aerospace Conference paper
investigated a lightweight PaDiM-based model for aero-turbine borescope
inspection, using MobileNetV3 as the backbone and evaluating
training/prediction time and detection performance.

This supports the research direction of replacing computationally
expensive feature extractors with lightweight backbones while
maintaining useful anomaly-detection performance.

**Reference:**\
["A Lightweight Anomaly Detection Model in Aero Turbine Borescope Using
Unsupervised Deep Learning," IEEE Aerospace Conference
2025](https://ieeexplore.ieee.org/document/11068398/)

------------------------------------------------------------------------

## 3.10 Knowledge Distillation for Lightweight Industrial Anomaly Detection

A 2025 IEEE ICIPCA paper investigated cross-architecture knowledge
distillation in which DINOv2-small acts as a teacher and a lightweight
ResNet18 acts as a student. The work explicitly targets industrial
anomaly detection and computational efficiency.

This suggests a promising direction for the proposed project: knowledge
transfer from a stronger feature representation to a smaller student
model can potentially reduce deployment cost while preserving detection
and localization performance.

**Reference:**\
["Exploring Cross-Architecture Distillation for Industrial Anomaly
Detection," IEEE ICIPCA
2025](https://ieeexplore.ieee.org/document/11138658/)

------------------------------------------------------------------------

# 11. Previously Published Research Papers

The following papers are recommended as the initial literature set. IEEE
publications are prioritized, while foundational CVPR/WACV papers are
included because several of the most influential industrial
anomaly-detection methods were published through the IEEE/CVF conference
ecosystem.

## IEEE-focused papers

### 1. A Comprehensive Review of Convolutional Neural Networks for Defect Detection in Industrial Applications

**Venue:** IEEE Access, 2024\
**DOI:** 10.1109/ACCESS.2024.3425166

**Relevance:** Provides a broad review of CNN-based industrial defect
detection, applications, hardware considerations and challenges such as
limited datasets and computational complexity.

**Access:**\
https://ieeexplore.ieee.org/document/10589380/

------------------------------------------------------------------------

### 2. A Review of Deep Learning-Based Anomaly Detection Strategies in Industry 4.0 Focused on Application Fields, Sensing Equipment, and Algorithms

**Venue:** IEEE Access, 2024\
**DOI:** 10.1109/ACCESS.2024.3424488

**Relevance:** Useful for establishing the broader anomaly-detection
literature, industrial applications and research directions.

**Access:**\
https://ieeexplore.ieee.org/document/10587235/

------------------------------------------------------------------------

### 3. Surface Defect Detection Methods Based on Deep Learning: a Brief Review

**Venue:** IEEE ITCA, 2020\
**DOI:** 10.1109/ITCA52113.2020.00049

**Relevance:** Useful for the traditional-to-deep-learning evolution of
industrial surface-defect detection.

**Access:**\
https://ieeexplore.ieee.org/document/9421960/

------------------------------------------------------------------------

### 4. Unsupervised Image Anomaly Detection and Localization in Industry Based on Self-Updated Memory and Center Clustering

**Venue:** IEEE Transactions on Instrumentation and Measurement, 2023\
**DOI:** 10.1109/TIM.2023.3271754

**Relevance:** Directly related to unsupervised industrial anomaly
detection, feature distributions, memory banks and localization.

**Access:**\
https://ieeexplore.ieee.org/document/10114470/

------------------------------------------------------------------------

### 5. Self-Supervised Masking for Unsupervised Anomaly Detection and Localization

**Venue:** IEEE Transactions on Multimedia, 2023\
**DOI:** 10.1109/TMM.2022.3175611

**Relevance:** Relevant to self-supervised anomaly detection and
pixel/region-level localization.

**Access:**\
https://ieeexplore.ieee.org/document/9779083/

------------------------------------------------------------------------

### 6. A Lightweight Anomaly Detection Model in Aero Turbine Borescope Using Unsupervised Deep Learning

**Venue:** IEEE Aerospace Conference, 2025\
**DOI:** 10.1109/AERO63441.2025.11068398

**Relevance:** Particularly relevant to the proposed lightweight
direction. The paper investigates a lightweight PaDiM-based system using
a MobileNetV3 backbone.

**Access:**\
https://ieeexplore.ieee.org/document/11068398/

------------------------------------------------------------------------

### 7. Exploring Cross-Architecture Distillation for Industrial Anomaly Detection

**Venue:** IEEE ICIPCA, 2025\
**DOI:** 10.1109/ICIPCA65645.2025.11138658

**Relevance:** Highly relevant to lightweight model design through
teacher-student knowledge distillation.

**Access:**\
https://ieeexplore.ieee.org/document/11138658/

------------------------------------------------------------------------

### 8. Enhancing Industrial PCB and PCBA Defect Detection: An Efficient and Accurate SEConv-YOLO Approach

**Venue:** IEEE Access, 2025\
**DOI:** 10.1109/ACCESS.2025.3601151

**Relevance:** Useful for the literature section on lightweight
architectures, efficient feature extraction and industrial defect
detection.

**Access:**\
https://ieeexplore.ieee.org/document/11132311/

------------------------------------------------------------------------

### 9. A Novel Fusion Attention-Based Lightweight Model for Pipeline Weld Multiscale Defect Detection

**Venue:** IEEE Transactions on Industrial Informatics, 2026\
**DOI:** 10.1109/TII.2026.3659675

**Relevance:** Useful recent work for lightweight industrial defect
detection, multi-scale features and resource-constrained deployment.

**Access:**\
https://ieeexplore.ieee.org/document/11393586/

------------------------------------------------------------------------

### 10. Towards High-Performance and Lightweight Anomaly Detection in Aero-Engine Blades Without Anomalous Examples

**Venue:** IEEE AIHCIR, 2025\
**DOI:** 10.1109/AIHCIR67580.2025.11405234

**Relevance:** Particularly relevant to normal-only training, anomaly
localization and lightweight industrial inspection.

**Access:**\
https://ieeexplore.ieee.org/document/11405234/

------------------------------------------------------------------------

## Foundational IEEE/CVF papers

### 11. MVTec AD --- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection

**Venue:** IEEE/CVF CVPR, 2019

**Relevance:** Foundational industrial anomaly-detection benchmark and
the recommended primary dataset for this project.

**Access:**\
https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html

------------------------------------------------------------------------

##

# 12. Recommended Literature-Survey Comparison Matrix

For the Phase-I literature survey, use a table similar to the following:

  ----------------------------------------------------------------------------------------------------------------------------------
  Paper                       Year Approach            Training    Localization   Accuracy   Computational    Limitation
                                                       Data                       Metric     Concern          
  -------------------- ----------- ------------------- ----------- -------------- ---------- ---------------- ----------------------
  MVTec AD                    2019 Dataset/benchmark   Normal +    Yes            AUROC      Benchmark        Not a model
                                                       anomalous                                              
                                                       test                                                   

  Surface Defect              2020 Survey              Various     Various        Various    Discussed        Review only
  Review                                                                                                      

  PaDiM                       2021 Feature             Normal      Yes            AUROC      Feature          Memory/feature cost
                                   distribution                                              computation      

  PatchCore                   2022 Memory bank         Normal      Yes            AUROC      Memory bank      Resource requirements

  Self-Supervised             2023 Self-supervised     Mostly      Yes            AUC        Reconstruction   Model complexity
  Masking                          reconstruction      normal                                cost             

  SMCC                        2023 Memory + clustering Normal      Yes            AUROC      Feature/memory   Dataset adaptation
                                                                                             cost             

  EfficientAD                 2024 Student-teacher +   Normal      Yes            AUROC      Designed for     Further edge
                                   AE                                                        efficiency       optimization possible

  Lightweight PaDiM           2025 PaDiM + MobileNetV3 Normal      Yes            AUROC/F1   Reduced cost     Domain-specific
                                                                                                              evaluation

  Cross-Architecture          2025 DINOv2 → ResNet18   Normal      Yes            AUROC      Student          Further
  Distillation                                                                               efficiency       pruning/quantization

  Proposed Work                --- Lightweight VAD     Normal      Yes            Multiple   **Target:        To be evaluated
                                                                                             reduced cost**   
  ----------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

---

## Reference verification status

| # | Paper | Status |
|---|---|---|
| 11 | MVTec AD (CVPR 2019) | Verified on the dataset owner's page (also lists IJCV 2021 extended version) |
| 12, 13 | PatchCore, EfficientAD | Well-known; URLs from the spec point to CVF open access |
| 4 (PaDiM), 3 (ITCA 2020), 1–2 (IEEE Access 2024) | — | Not re-checked here; confirm DOI/pages on IEEE Xplore before citing |
| 5, 6, 7, 8, 9, 10 | IEEE 2023–2026 papers | ⚠ **Not independently verified.** A web search for paper 10's exact title returned no matching record (this does not prove it is wrong - IEEE Xplore is not always indexed). Open every IEEE link, confirm title/authors/venue/year/DOI, and drop any you cannot find. |

Never cite a paper in the survey that you have not opened yourself.
