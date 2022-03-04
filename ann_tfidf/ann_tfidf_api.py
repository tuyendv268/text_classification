# %%
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import torch.nn as nn
import torch
import numpy as np
from pyvi import ViTokenizer

with open("C:\\Users\\tuyen.dv\\Documents\\NLP\\NLP_tuan_3\\Data\\stopword.txt", encoding="utf-8") as f:
  stopwords = f.readlines()
stopwords = [x.strip().replace(" ", "_") for x in stopwords]
with open("C:\\Users\\tuyen.dv\\Documents\\NLP\\NLP_tuan_3\\Data\\datasetVNTC.json", encoding="utf-8") as f:
  data_train = json.load(f)

tfidf_vector = TfidfVectorizer(stop_words=stopwords)
tfidf_matrix = tfidf_vector.fit_transform(data_train['data'], data_train['target'])

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
svd = TruncatedSVD(n_components=300)
svd.fit(tfidf_matrix)
model = MultilayerPerceptron(300, 10)
model.load_state_dict(torch.load('ann_tfidf.pth'))

new_data = "Theo Reuters, lãnh đạo các quốc gia Baltic và Trung Âu hôm 16-12 nói rằng EU đang bị Nga tấn công từ nhiều mặt và phải đoàn kết sau các biện pháp trừng phạt kinh tế mới, trong đó Lithuania viện dẫn nguy cơ có thể bị Nga tấn công quân sự từ Belarus."

tokenized_new_data = ViTokenizer.tokenize(new_data)
input_data_preprocessed = tfidf_vector.transform([tokenized_new_data])
svd_tfidf_vector = svd.transform(input_data_preprocessed)

result =  np.argmax(model(torch.tensor(svd_tfidf_vector, dtype = torch.float)).tolist() ,axis = 1)

print("data : ", new_data)
print("predict : " ,data_train['target_names'][result[0]])


