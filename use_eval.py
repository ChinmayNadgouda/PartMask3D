from eval import main
import numpy as np

def load_data_from_npy(path):
    return np.load(path)

#training
hook_pull = '2b0fed8b-f58e-4ffb-9e4f-ccdd880817c0.npy'
train_ids = ['0bfc68dc-c9c4-45d2-9dbf-b58198d21f8d.npy', '0ac7dd02-01df-451b-a714-0980f9183d11.npy', '0a1daa04-d089-46dc-b752-f2a32759dd5b.npy']
working_pull = '1d366253-dfa4-4466-8be6-f3e5a032b370.npy'

#testing
test = ['00b40f17-72f8-40c2-a374-ea1959499c9f.npy', 'ed98f6d9-ddb8-4f76-93ed-95c2801931e9.npy', 'e32b1215-8cfe-45e6-b031-cd5a5297d7f1.npy','8e56f208-6939-4571-85b8-8f53c58fcbe3.npy']
croos_attnError = '003f76e0-9384-478c-839f-b54a0ce3f229.npy'
input_data2 = load_data_from_npy('/home/gokul/ConceptGraphs/concept-graphs/conceptgraph/Mask3D/data/processed/scannet200/train/0ac7dd02-01df-451b-a714-0980f9183d11.npy') 
from hydra.experimental import initialize, compose
initialize(config_path="conf", job_name="test_app")  # Initialize Hydra
cfg = compose(config_name="config_base_instance_segmentation.yaml")  # Load the Hydra configuration
main(cfg, input_data2[:,:3], input_data2[:,3:6], input_data2[:,6:9])