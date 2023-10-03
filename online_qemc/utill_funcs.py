"""TODO COMPLETE."""


from typing import Dict

def prob_dist_to_partition_bitstring(
    prob_dist: Dict[str, float],
    num_nodes_to_count: int,
    num_nodes_total: int
) -> str:
    """TODO COMPLETE."""
    
    bitstring = ["0" for _ in range(num_nodes_total)]
    sorted_prob_dist_items = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)
    
    for index, prob_item in enumerate(sorted_prob_dist_items):
        bitstring[int(prob_item[0], 2)] = "1"
        
        if index == num_nodes_to_count - 1:
            break
            
    return "".join(bitstring)