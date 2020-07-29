import torch 
import numpy as np 



def answer_question(tokenizer, start_position, end_position,ignore_index,c_tokens, k=20):
    topk_start_position = torch.topk(start_position, k=k)
    topk_end_position = torch.topk(end_position,k=k)

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
                    if len(top20) < k:
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
        answer = tokenizer.convert_tokens_to_string(c_tokens[(final_start_index-ignore_index-1): (final_end_index-ignore_index)])
        print(answer) 