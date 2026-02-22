import torch

def mask_from_id_to_id(input_seq,mask_start_id=9,mask_end_id=10):
    mask = torch.zeros_like(input_seq, dtype=torch.bool)
    answer_positions = (input_seq==mask_start_id).nonzero(as_tuple=True)[0]
    
    if len(answer_positions) > 0:
        for start_pos in answer_positions:
            end_positions = (input_seq[start_pos:] == mask_end_id).nonzero(as_tuple=True)[0]
            if len(end_positions) > 0:
                end_pos = start_pos + end_positions[0]
                mask[start_pos:end_pos] = True
    return mask