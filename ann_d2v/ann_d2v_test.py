
# %%
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import gensim
import json
from gensim.models import Doc2Vec 
from pyvi import ViTokenizer

# %%
d2v_model = Doc2Vec.load("C:\\Users\\tuyen.dv\\Documents\\NLP\\NLP_tuan_3\\Ann_d2v\\doc2vec.model")

# %%
with open("C:\\Users\\tuyen.dv\\Documents\\NLP\\NLP_tuan_3\\Data\\stopword.txt", encoding="utf-8") as f:
  stopwords = f.readlines()
stopwords = [x.strip().replace(" ", "_") for x in stopwords]

# %%
with open("C:\\Users\\tuyen.dv\\Documents\\NLP\\NLP_tuan_3\\Data\\datasetVNTC.json", encoding="utf-8") as f:
  data_train = json.load(f)

# %%
class MultilayerPerceptron(nn.Module):

  def __init__(self, input_size,output_size):
    super(MultilayerPerceptron, self).__init__()

    # Saving the initialization parameters
    self.input_size = input_size 
    self.output_size = output_size

    # Defining model
    self.model = nn.Sequential(
        nn.Linear(self.input_size, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, output_size),
    )
    
  def forward(self, x):
    output = self.model(x)
    return output

# %%
model = MultilayerPerceptron(300, 10)

# %%
model.load_state_dict(torch.load('C:\\Users\\tuyen.dv\\Documents\\NLP\\NLP_tuan_3\\Ann_d2v\\ann_d2v1.pth'))

# %%
new_data = "Đến hôm qua, trận đấu giữa Burnley và Watford đã phải hoãn lại chỉ 2 tiếng rưỡi trước giờ bóng lăn. Lý do vì một số lượng lớn cầu thủ Watford có kết quả dương tính với SARS-CoV-2. Hôm nay, trận đấu thứ 3 tại vòng 17 Ngoại hạng Anh, Leicester vs Tottenham cũng chính thức bị hoãn."

# %%
tokenized_new_data = ViTokenizer.tokenize(new_data)

# %%
temp = [word for word in tokenized_new_data.split() if word not in stopwords]

# %%
data_preprocessed = gensim.utils.simple_preprocess(' '.join(temp))

# %%
X = d2v_model.infer_vector(data_preprocessed)

# %%
print(data_train['target_names'][np.argmax(model(torch.tensor(X, dtype = torch.float)).tolist())])


