<h1 align="center"><strong>FineXtrol: Controllable Motion Generation via Fine-Grained Text</strong></h1>
  <p align="center">
    <a href='https://scholar.google.com/citations?hl=zh-CN&authuser=1&user=e5YyAxcAAAAJ' target='_blank'>Keming Shen</a><sup>1,2</sup>&emsp;
    Bizhu Wu</a><sup>1,2,3</sup>&emsp;
    Junliang Chen</a><sup>4</sup>&emsp;
    Xiaoqin Wang</a><sup>1,2</sup>&emsp;
    Linlin Shen</a>*<sup>1,2,5</sup>&emsp;
    <br>
    <sup>1</sup>School of Computer Science and Software Engineering, Shenzhen University<br>
    <sup>2</sup>Guangdong Provincial Key Laboratory of Intelligent Information Processing<br>
    <sup>3</sup>University of Nottingham Ningbo China&emsp;
    <sup>4</sup>The Hong Kong Polytechnic University<br>
    <sup>5</sup>Computer Vision Institute, School of Artificial Intelligence, Shenzhen University
    <br>
    *Indicates Corresponding Author
</p>
</p>

<p align="center">
  <!-- <a href="https://neurips.cc/virtual/2025/poster/118773">
    <img src="https://img.shields.io/badge/AAAI-2026-red
    " alt="AAAI 2026">
  </a> -->
  <a href="https://arxiv.org/abs/2511.18927">
    <img src="https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow" alt="Paper PDF on arXiv">
  </a>
  <a href="https://lucky1192405754.github.io/FineXtrol/">
    <img src="https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green" alt="Project Page">
  </a>
</p>

</div>
How to achieve **precise** spatial-temporal control in motion generation using only **natural language**?

> **TL;DR:** We propose **FineXtrol**, a novel control framework that utilizes *fine-grained textual control signals* (e.g., "Straighten your left leg in 1.0-1.5s") to guide motion generation. By combining a **Dual-Branch ControlNet** architecture with a **Hierarchical Contrastive Learning** module, FineXtrol enables precise control over specific body parts within designated temporal intervals, offering a more user-friendly and efficient alternative to spatial coordinate-based methods.

<div align="center">
    <img src="static/images/pipeline.png" alt="FineXtrol Pipeline" class="blend-img-background center-image" style="max-width: 100%; height: auto;" />
</div>

## 📣 News
- **[2025/11]** Our paper "FineXtrol" has been accepted by **AAAI 2026**! 🎉
- **[2025/11]** The [Project Page](https://lucky1192405754.github.io/FineXtrol/) is now live.
- **[2025/11]** The paper is available on [ArXiv](https://arxiv.org/abs/2511.18927).

## 📆 Plan
- [x] Release paper and project page.
- [ ] Release the **FineMotion** dataset processing scripts.
- [ ] Release **FineXtrol** code:
  - [ ] Environment setup guidance.
  - [ ] Inference scripts (Pretrained models).
  - [ ] Training scripts (Hierarchical Contrastive Learning & Diffusion).
  - [ ] Evaluation metrics.

## 🛠️ Installation & Usage
*(Coming Soon...)*

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- https://github.com/GuyTevet/motion-diffusion-model
- https://github.com/BizhuWu/FineMotion
- https://github.com/neu-vi/OmniControl
- https://github.com/Dai-Wenxun/MotionLCM

<!-- ## Citation
If you find this repository/work helpful in your research, please consider citing the paper and starring the repo ⭐.

```
@article{tan2025sopo,
  title={SoPo: Text-to-Motion Generation Using Semi-Online Preference Optimization},
  author={Tan, Xiaofeng and Wang, Hongsong and Geng, Xin and Zhou, Pan},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
``` -->
