import requests
import os
import torch

class IPFSHandler:
    def __init__(self, api_url="http://127.0.0.1:5001/api/v0"):
        self.api_url = api_url
        self.temp_dir = "./temp_models"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def upload_model(self, model_state_dict, filename):
        """
        Saves the state_dict to a local file, then uploads to IPFS.
        Returns: The IPFS CID (Hash) as a string.
        """
        # 1. Save to local disk
        filepath = os.path.join(self.temp_dir, filename)
        torch.save(model_state_dict, filepath)

        # 2. Upload to IPFS via HTTP API
        # Equivalent to CLI: ipfs add <file>
        try:
            with open(filepath, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.api_url}/add", files=files)
                response.raise_for_status()
                
                # Parse CID from response
                # Response format: {"Name":"...","Hash":"Qm...","Size":"..."}
                data = response.json()
                return data['Hash']
        except Exception as e:
            print(f"❌ IPFS Upload Failed: {e}")
            return None

    def download_model(self, cid):
        """
        Downloads a file from IPFS given a CID and loads it as a state_dict.
        """
        try:
            # Equivalent to CLI: ipfs cat <cid>
            params = {'arg': cid}
            response = requests.post(f"{self.api_url}/cat", params=params)
            response.raise_for_status()

            # Save binary content to temp file
            download_path = os.path.join(self.temp_dir, f"downloaded_{cid}.pth")
            with open(download_path, 'wb') as f:
                f.write(response.content)

            # Load back into PyTorch
            state_dict = torch.load(download_path)
            return state_dict
        except Exception as e:
            print(f"❌ IPFS Download Failed: {e}")
            return None

    def clear_temp_files(self):
        """Cleanup local storage"""
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))