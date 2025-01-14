from eval import main
import numpy as np

def load_data_from_npy(path):
    return np.load(path)

input_data2 = load_data_from_npy('/home/gokul/ConceptGraphs/concept-graphs/conceptgraph/Mask3D/data/processed/scannet200/test/277a29e0-a959-4a41-8cd8-327ac427c499.npy')
from hydra.experimental import initialize, compose
initialize(config_path="conf", job_name="test_app")  # Initialize Hydra
cfg = compose(config_name="config_base_instance_segmentation.yaml")  # Load the Hydra configuration
main(cfg, input_data2[:,:3], input_data2[:,3:6], input_data2[:,6:9])