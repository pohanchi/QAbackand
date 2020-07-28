from examples.transformers import AlbertForQuestionAnswering
from examples.transformers import BertTokenizer
from examples.transformers import AlbertConfig
import torch
import IPython
import pdb 
import numpy as np

devices = torch.device("cuda")
config = AlbertConfig.from_pretrained("output/FirstModel")
tokenizer = BertTokenizer.from_pretrained("output/FirstModel")
model =AlbertForQuestionAnswering.from_pretrained("output/FirstModel").to(devices)

context = "國立交通大學，簡稱交大，原建於上海市，後復校於新竹市，為中華民國一所研究型大學。該校主要目的為培育工程、科學及管理方面的人才，此宗旨現於交大校徽上的E、S、A。國立交通大學前身為1896年由盛宣懷創立於上海市徐家匯的南洋公學。在中國抗日戰爭中經歷多次遷校及改組。於國共內戰後，上海原址改組為上海交通大學，並於1958年由教育部選定新竹市為交通大學復校後校址，復校後校址與新竹科學工業園區及國立清華大學相鄰。今日的國立交通大學，為邁向頂尖大學計畫成員，主要發展領域為電子、資通訊及光電等，為臺灣知名院校之一，曾一度與國立清華大學洽談合併事宜，但因新校名稱問題而破局。位於新竹市的交通大學也同上海交通大學、西安交通大學、西南交通大學、北京交通大學並稱「飲水思源 五校一家」，代表五校皆系出於同源。飲水思源紀念碑也為各校的精神團結的象徵之一。"

c_tokens = tokenizer.tokenize(context)
context_ids = tokenizer.convert_tokens_to_ids(c_tokens)

question = "國立交通大學與哪間大學相鄰"

q_tokens = tokenizer.tokenize(question)
question_ids = tokenizer.convert_tokens_to_ids(q_tokens)

extra_ids = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", "<pad>"])

input_ids = torch.tensor([[extra_ids[0]]+question_ids+[extra_ids[1]]+context_ids+[extra_ids[1]]])
segment_ids = torch.tensor([[0]+[0]*len(question_ids)+[1]+[1]*len(context_ids)+[1]])
attention_mask = torch.tensor([[1]*input_ids.shape[0]])
ignore_index = 1 + len(question_ids) + 1 - 1

# print(extra_ids)
# print(attention_mask)
# print(input_ids)
# print(segment_ids)
# print(ignore_index)

input_dict = {"input_ids":input_ids.cuda(), "attention_mask":attention_mask.cuda(), "token_type_ids":segment_ids.cuda(), "position_ids":None, "head_mask":None,"inputs_embeds": None, "start_positions":None, "end_positions":None}

start_position, end_position=model(**input_dict)
# print(start_position, end_position)

k=20
topk_start_position = torch.topk(start_position, k=20)
topk_end_position = torch.topk(end_position,k=20)

condition_start = ((topk_start_position[1] >= ignore_index) | (topk_start_position[1] ==0) )
condition_end = ((topk_end_position[1] >= ignore_index) | (topk_end_position[1]==0) )

filter_start_index = topk_start_position[1][condition_start]
filter_end_index = topk_end_position[1][condition_end]
filter_start_logits = topk_start_position[0][condition_start]
filter_end_logits = topk_end_position[0][condition_end]

start_length = filter_start_index.shape[0]
end_length = filter_end_index.shape[0]
max_answer_length = 30

top20 = []
for i in range(start_length):
    for j in range(end_length):
        if filter_start_index[i] == 0 and filter_end_index[j] == 0:
            null_logits = filter_start_logits[i] + filter_end_logits[j]
        else:
            if filter_start_index[i] == 0:
                continue
            if filter_end_index[j] - filter_start_index[i] < max_answer_length:
                logits=filter_start_logits[i] + filter_end_logits[j]
                if len(top20) < 20:
                    top20.append([filter_start_index[i], filter_end_index[j], logits])
                else:
                    tmp_array=np.array(top20)
                    index=np.argmin(tmp_array[:,2])
                    if logits > tmp_array[index,2]:
                        top20[index] = [filter_start_index[i], filter_end_index[j], logits]
                    else:
                        pass

result = np.array(top20)
index = np.argmax(result[:,2])
final_start_index, final_end_index, final_logits = top20[index]

if final_logits < null_logits: 
    print("不能回答")
else: 
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0,final_start_index: final_end_index+1].tolist()))
    print(answer) 




