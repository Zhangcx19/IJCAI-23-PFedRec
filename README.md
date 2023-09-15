# Dual Personalization on Federated Recommendation
Code for ijcai-23 paper: [Dual Personalization on Federated Recommendation](https://www.ijcai.org/proceedings/2023/0507.pdf).

## Abatract
Federated recommendation is a new Internet service architecture that aims to provide privacypreserving recommendation services in federated settings. Existing solutions are used to combine distributed recommendation algorithms and privacy-preserving mechanisms. Thus it inherently takes the form of heavyweight models at the server and hinders the deployment of on-device intelligent models to end-users. This paper proposes a novel Personalized Federated Recommendation (PFedRec) framework to learn many user-specifc lightweight models to be deployed on smart devices rather than a heavyweight model on a server. Moreover, we propose a new dual personalization mechanism to effectively learn fne-grained personalization on both users and items. The overall learning process is formulated into a unifed federated optimization framework. Specifcally, unlike previous methods that share exactly the same item embeddings across users in a federated system, dual personalization allows mild fnetuning of item embeddings for each user to generate user-specifc views for item representations which can be integrated
into existing federated recommendation methods to gain improvements immediately. Experiments on multiple benchmark datasets have demonstrated the effectiveness of PFedRec and the dual personalization mechanism. Moreover, we provide visualizations and in-depth analysis of the personalization techniques in item embedding, which shed novel insights on the design of recommender systems in federated settings.

![](https://github.com/Zhangcx19/IJCAI-23-PFedRec/blob/main/architecture%20comparison.png)
**Figure:**
Different frameworks for the personalized federated recommendation. The green block represents a personalized module, which indicates the part of model is to preserve user preference. Our proposed model will preserve dual personalization on two modules.

## Preparations before running the code
mkdir log

mkdir sh_result

## Running the code
python train.py

## Citation
If you find this project helpful, please consider to cite the following paper:

```
@article{zhang2023dual,
  title={Dual Personalization on Federated Recommendation},
  author={Zhang, Chunxu and Long, Guodong and Zhou, Tianyi and Yan, Peng and Zhang, Zijian and Zhang, Chengqi and Yang, Bo},
  journal={arXiv preprint arXiv:2301.08143},
  year={2023}
}
```
