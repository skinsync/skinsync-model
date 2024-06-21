# skinsync_model-Skin Type Classification
## Team ID: C241-PS124

### ML Team Members:
- (ML) M335D4KY2083 – Gybran Khairul Anam – Universitas Yarsi - [Active]
- (ML) M332D4KX1670 – Cheryl Almirah Azmi – Universitas Trunojoyo - [Active]
- (ML) M204D4KY2509 – Naufal Hafizh Muttaqin – Universitas Esa Unggul - [Active]

## 1. Dataset
The dataset used consists of images of different skin types and facial skin problems. The dataset is divided into two main categories: "Skin Type Classification" and "Skincare Recommendation."

### Skin Type Classification
This dataset contains images of faces with different skin types, such as:
- Normal 😊
- Oily 💦
- Dry 🌵
- Acne 😓

Kaggle Dataset = https://www.kaggle.com/datasets/muttaqin1113/face-skin-type

The graphs below illustrate the distribution of various skin types within our dataset. This visualization helps us to ensure that our dataset is balanced and provides insights into the prevalence of each skin type, which is essential for training robust machine learning models. A balanced dataset allows the model to learn equally from each category, thereby enhancing its ability to generalize well to unseen data. The following bar graph and pie chart showcase the detailed breakdown of the dataset used in our skin type classification project.

![bar graph](https://cdn.discordapp.com/attachments/1200427587940392991/1253651700594511944/rABdpn0AAACRixQAAEAkpAAAACIhBQAAEAkpAACASEgBAABEQgoAACASUgAAAJGQAgAAiD4ymU8DB3Rt5wAAAABJRU5ErkJggg.png?ex=6676a1a1&is=66755021&hm=78cee51f725fb871631e294d25271993dad5f2d01baed89b600e33b51ee76622&) ![pie chart](https://cdn.discordapp.com/attachments/1200427587940392991/1253651667975405580/w8elyhbdNOfYgAAAABJRU5ErkJggg.png?ex=6676a199&is=66755019&hm=51edae12b36970ed29c4b6e1b84a25fb6bbebba88b40422b1fa15dd3abe8d5d5&)


### Skincare Recommendation
This dataset includes information on various skincare products and their suitability for different skin types and conditions.

Github = https://github.com/Yunanouv/Skin-Care-Recommender-System/blob/main/MP-Skin%20Care%20Product%20Recommendation%20System3.csv


## 2. Methods
We employed a Convolutional Neural Network (CNN) model to classify skin types and provide skincare recommendations based on the classification results. The methodology includes data preprocessing, model training, and evaluation.
![Methods](https://cdn.discordapp.com/attachments/1200427587940392991/1253651308683071489/Frame_15299.png?ex=6676a143&is=66754fc3&hm=d324614970a96964660cf406e0e293bc80438ac90712c4288d19173404e98180&)


## 4. Experimental Design
![Experimental Design](https://i.sstatic.net/osBuF.png)


## 5. Model Architecture
A Convolutional Neural Network (CNN) model is used to classify the dataset categories. The model architecture can be customized based on requirements, but for this project, the following architecture is used:
- **Input Layer**
- **Convolutional Layers**: Used for feature extraction from the images.
- **Max Pooling Layers**: Used for dimensionality reduction of the features.
- **Flatten Layer**: Flattens the features into a vector.
- **Fully Connected Layers**: Perform classification tasks.
- **Output Layer**: Outputs the classification predictions based on labels.

## 4. Results
The model was evaluated using a testing generator, resulting in the following accuracy and loss metrics:

| Metric       | Value   |
|--------------|---------|
| Test Loss    | 0.3207  |
| Test Accuracy| 0.9034  |

## 5. Requirements
- **Python**
- **Keras** 
- **TensorFlow** 
- **NumPy** 
- **Matplotlib** 
- **sckit-learn**
- **pandas**
- **seaborn**


## 7. References
1. Convolutional Neural Networks (CNNs) - Stanford University, https://cs231n.github.io/convolutional-networks/
3. Dataset:
   - https://www.kaggle.com/datasets/muttaqin1113/face-skin-type
   - https://github.com/Yunanouv/Skin-Care-Recommender-System/blob/main/MP-Skin%20Care%20Product%20Recommendation%20System3.csv
4. Paper for Pretrained Model:
   - Howard, A. G. (2017, April 17). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv.org. https://arxiv.org/abs/1704.04861




















