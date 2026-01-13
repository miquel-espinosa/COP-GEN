import torch
import os
import argparse

def extract_ema_to_pth(pt_path):
    data = torch.load(pt_path, map_location='cpu')
    
    # Extract EMA state dict
    state_dict = data['ema']
    # Keep only ema_model keys (remove keys starting with 'online_model.' or 'loss.')
    processed_state_dict = {
        k[len("ema_model."):]: v for k, v in state_dict.items() 
        if k.startswith("ema_model.") and not k.startswith("online_model.")
    }
    
    # Remove any loss-related keys
    processed_state_dict = {k: v for k, v in processed_state_dict.items() if not k.startswith("loss.")}
    
    # Save processed state dict
    output_path = os.path.join(os.path.dirname(pt_path), 
                              os.path.basename(pt_path).replace('.pt', '-ema.pth'))
    torch.save(processed_state_dict, output_path)
    print(f"Saved processed EMA model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and process EMA model from checkpoint file")
    parser.add_argument("model_path", type=str, help="Path to the checkpoint file containing EMA model")
    args = parser.parse_args()
    
    extract_ema_to_pth(args.model_path)
