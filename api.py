import flask
import json
import requests
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
import math
import torch

# from tfidf import keywords_byID

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

f = open("titles.txt", "r")
documents = f.read().split('\n')

def cos_similarity(X, B):
  List = []
  B = B.toarray().reshape(-1)
  for idx in range(X.shape[0]):
    A = X[idx].toarray().reshape(-1)
    cos_sim=np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B)) * len(documents[idx])
    List.append(cos_sim)
  return List


@app.route('/', methods=['GET'])
def home():
    return "<h1>Hello Flask!</h1>"

@app.route('/search/<name>', methods=['GET'])
def search(name):
  score = title_to_score(name)
  
  return str(score)

@app.route('/article/<test_data>', methods=['GET'])
def article(test_data):
  documents.append(test_data)
  last = len(documents) - 1
  tfidf = TfidfVectorizer().fit_transform(documents)

  a = cosine_similarity(tfidf[:last], tfidf[last:])
  a2 = np.array([a[idx][0] * len(documents[idx]) for idx in range(len(a))])
  ans = heapq.nlargest(3, range(len(a2)), a2.take)
  
  codes = []
  for idx in ans:
    codes.append(documents[idx])
  documents.remove(test_data)
  res = { 'data': codes }
  return json.dumps(res)

requests.packages.urllib3.disable_warnings()
cookie = '__auc=6dc9ccdd17143b13780f7882a84; __gads=ID=4a1a6db174d6b764:T=1585978677:S=ALNI_MYhw2OFmwPMXG1d6oDGLtOSaoaUBw; _gid=GA1.2.1722273152.1586581391; dcard=eyJ0b2tlbiI6bnVsbCwiX2V4cGlyZSI6MTU4OTE3NTA2NjgzNCwiX21heEFnZSI6MjU5MjAwMDAwMH0=; dcard.sig=V23CgwwWlQwERJ_4oClxCfx3eJo; dcsrd=u5srf5XIcfZBSFOWoTUlB9A1; dcsrd.sig=ILdWwvOz55JmB6jAgNThu9LMRwA; __asc=7bc8218c171681f2a3a6d824441; _gat=1; amplitude_id_bfb03c251442ee56c07f655053aba20fdcard.tw=eyJkZXZpY2VJZCI6ImU0Njc2NzQ5LTQyZWUtNDFhNS04ZWE4LTA5N2JkMzhhOWZkMlIiLCJ1c2VySWQiOiI1OTk2MDgxIiwib3B0T3V0IjpmYWxzZSwic2Vzc2lvbklkIjoxNTg2NTg5ODA1ODU3LCJsYXN0RXZlbnRUaW1lIjoxNTg2NTg5ODIxMTk2LCJldmVudElkIjozNiwiaWRlbnRpZnlJZCI6MjIsInNlcXVlbmNlTnVtYmVyIjo1OH0=; __cfduid=d9fff663f1a6dd3065426909f15752fcb1586589920; _ga=GA1.1.2128013235.1585978618; _ga_C3J49QFLW7=GS1.1.1586589804.10.1.1586589846.0'
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
    'cookie': cookie,
    # 'X-CSRF-Token': token,
    'Connection':'close'
}
tfidf_post_keys = ['title']
tfidf_comment_keys = ['content']
dcard_api = 'https://www.dcard.tw/service/api/v2/'
retries = 3

def url_to_json(url):
  for i in range(retries):
    response = requests.get(url, headers=headers, verify=False)
    if response.status_code == requests.codes.ok:
      json_file = response.json()
      response.close()
      return json_file
    elif response.status_code == 404:
      return None
    else:
      print('status error:',  response.status_code)
      continue
      # response.raise_for_status()
    response.close()
  print('EXCEED RETRIES: ', url)

if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load a trained model and vocabulary that you have fine-tuned
load_pretrain_dir = './model_save/'
model = BertForSequenceClassification.from_pretrained(load_pretrain_dir)
tokenizer = BertTokenizer.from_pretrained(load_pretrain_dir)

# Copy the model to the GPU.
model.to(device)

max_length = 32

def title_to_score(title):
    encoded_dict = tokenizer.encode_plus(
                        title,                           # Sentence to encode.
                        add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                        max_length = max_length,        # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',          # Return pytorch tensors.
                   )
    
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    b_input_ids = input_ids.to(device)
    b_input_mask = attention_masks.to(device)

    model.eval()
    outputs = model(b_input_ids, token_type_ids=None, 
                        attention_mask=b_input_mask)
    logits = outputs[0]
    k = logits[0][1]
    score = round(math.exp(k) / (1 + math.exp(k)) * 100, 2)
    return score


app.run()