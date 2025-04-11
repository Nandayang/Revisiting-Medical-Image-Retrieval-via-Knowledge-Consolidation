# Revisiting-Medical-Image-Retrieval-via-Knowledge-Consolidation
This is the official repository for Anomaly-aware Content-based Image Recommendation (ACIR)

![image](https://github.com/Nandayang/Revisiting-Medical-Image-Retrieval-via-Knowledge-Consolidation/blob/main/repo_arxiv/abstractFig_00.png)

For more information about ACIR, please read the [following paper](https://www.sciencedirect.com/science/article/pii/S1361841525001008), [arxiv version](https://arxiv.org/pdf/2503.09370):  

Nan, Y., Zhou, H., Xing, X., Papanastasiou, G., Zhu, L., Gao, Z., ... & Yang, G. (2025). Revisiting medical image retrieval via knowledge consolidation. Medical Image Analysis, 103553.

Please cite this paper if you are using it for your research.

ACIR can better cluster images with appropriate cluster centers as well as identifying OOD samples.
![image](https://github.com/Nandayang/Revisiting-Medical-Image-Retrieval-via-Knowledge-Consolidation/blob/main/repo_arxiv/Fig4_new_00.png)

To achieve this, the network should be trained in two stages:
  1. Train the model without the reconstruction decoder.
  2. Enable the reconstruction decoder and freeze the weights of the remaining parts of ACIR.

## Citation

If you find this study useful in your research, please cite:

```bibtex
@article{nan2025revisiting,
  title={Revisiting medical image retrieval via knowledge consolidation},
  author={Nan, Yang and Zhou, Huichi and Xing, Xiaodan and Papanastasiou, Giorgos and Zhu, Lei and Gao, Zhifan and Frangi, Alejandro F and Yang, Guang},
  journal={Medical Image Analysis},
  pages={103553},
  year={2025},
  publisher={Elsevier}
}
