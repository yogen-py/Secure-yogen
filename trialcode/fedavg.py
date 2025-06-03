import torch
import logging

logger = logging.getLogger(__name__)

def fed_avg(model_list):
    """
    Implements Federated Averaging (FedAvg) algorithm
    Args:
        model_list: List of model state dictionaries
    Returns:
        Averaged model state dictionary
    """
    if not model_list:
        raise ValueError("Empty model list for federation")
    
    try:
        # Initialize with first model's keys
        averaged_dict = {}
        for key in model_list[0].keys():
            # Stack same parameters from all models and average them
            averaged_dict[key] = torch.stack([model[key] for model in model_list]).mean(0)
        
        logger.info(f"Successfully averaged {len(model_list)} models")
        return averaged_dict
    except Exception as e:
        logger.error(f"Error during model averaging: {str(e)}")
        raise

