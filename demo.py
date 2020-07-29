from examples import AlbertForQuestionAnswering
from examples import AlbertConfig
import torch
import IPython
import pdb 
import numpy as np

devices = torch.device("cuda")
config = AlbertConfig.from_pretrained("output/FirstModel")
tokenizer = BertTokenizer.from_pretrained("output/FirstModel")
model =AlbertForQuestionAnswering.from_pretrained("output/FirstModel").to(devices)

context = "在第二屆全走光後，很快的又到了推甄找教授的時期，彷彿不受到前一屆空無一人的情況，順利地收了三位推甄生，其中兩位為中山在校生，即使聽過許多流言蜚語，卻依然想要挑戰自我的進入了。而另一位為外校生，即使在四虎將不斷地暗示下，也依然堅持己見。並且在二月時，先進入實驗室開始地獄生活。這時外校推甄進來的這位，因為前一屆已經沒有人手，因此在學業上不斷地被給予壓力，希望能盡快有所產出。也因此學業進度及七圈精神不斷地被要求改進，想當然也被扣了不少未來薪水。並且老師還無法記取上一屆的教訓，持續威脅學生"

c_tokens = tokenizer.tokenize(context)
context_ids = tokenizer.convert_tokens_to_ids(c_tokens)

question = "老師收了幾位推甄生？"

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




