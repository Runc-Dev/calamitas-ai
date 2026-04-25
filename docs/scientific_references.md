# Scientific References

All techniques used in AFETSONAR with their original citations.

## Machine Learning & Computer Vision

| Ref | Citation | Used for |
|-----|----------|----------|
| Xie et al. 2021 | SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. *NeurIPS 2021*. arXiv:2105.15203. | Backbone encoder |
| Hinton et al. 2015 | Distilling the Knowledge in a Neural Network. arXiv:1503.02531. | Knowledge Distillation framework |
| Berman et al. 2018 | The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks. *CVPR 2018*. arXiv:1705.08790. | Lovász-Softmax loss |
| Lin et al. 2017 | Focal Loss for Dense Object Detection. *ICCV 2017*. arXiv:1708.02002. | Focal loss |
| Milletari et al. 2016 | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. *3DV 2016*. | Dice loss |
| Furlanello et al. 2018 | Born Again Networks. *ICML 2018*. | Knowledge Distillation design |
| Zhao et al. 2017 | Pyramid Scene Parsing Network. *CVPR 2017*. | Deep supervision pattern |
| Zagoruyko & Komodakis 2017 | Paying More Attention to Attention. *ICLR 2017*. | Attention transfer KD component |
| Loshchilov & Hutter 2017 | SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR 2017*. | Cosine warm restarts |

## Dataset

| Ref | Citation |
|-----|----------|
| Gupta et al. 2019 | xBD: A Dataset for Assessing Building Damage from Satellite Imagery. arXiv:1911.09296. |

## Routing & Graph Algorithms

| Ref | Citation | Used for |
|-----|----------|----------|
| Hart, Nilsson & Raphael 1968 | A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE TSSC* 4(2):100-107. | A* search |
| Rosenkrantz, Stearns & Lewis 1977 | An Analysis of Several Heuristics for the Traveling Salesman Problem. *SIAM J. Comput.* 6(3):563-581. | TSP nearest-neighbour |
| Yen 1971 | Finding the K Shortest Loopless Paths in a Network. *Management Science* 17(11):712-716. | k-shortest paths |
| Voronoi 1908 | Nouvelles applications des paramètres continus à la théorie des formes quadratiques. *J. Reine Angew. Math.* | Voronoi diagrams |
| MacQueen 1967 | Some methods for classification and analysis of multivariate observations. *5th Berkeley Symp.* | K-means clustering |

## Geographic / Geospatial

| Ref | Citation | Used for |
|-----|----------|----------|
| Sinnott 1984 | Virtues of the Haversine. *Sky & Telescope* 68(2):158. | Haversine distance formula |
| Pix4D Knowledge Base | Ground Sampling Distance (GSD) methodology. | Drone GSD calculation |

## Emergency Management Standards

| Ref | Citation | Used for |
|-----|----------|----------|
| FEMA P-154 (2015) | Rapid Visual Screening of Buildings for Potential Seismic Hazards. | Priority scoring |
| FEMA P-1070 (2016) | USAR Manual — Urban Search and Rescue. | Survival curve model |
| NATO STANAG 3204 (4th ed.) | Minimum Standards for Helicopter Landing Zones. | LZ dimension constraints |
| ICAO Annex 2 | Rules of the Air — Drone altitude limit 120 m AGL in uncontrolled airspace. | UAV altitude constant |
| AFAD HHTK 2019 | Hızlı Hasar Tespit Kriterleri (Turkish Rapid Damage Assessment). | Damage factor calibration |
| TÜİK 2023 | Türkiye İstatistik Kurumu — İstanbul İl Nüfus Yoğunluğu. | Population density constant |
