"""
CONSTANTS
"""

ROOT_PATH = "D:/TCM-MHNet/"
OUT_PATH = ROOT_PATH + "out/"
DATA_PATH = ROOT_PATH + "data/"
LOG_PATH = ROOT_PATH + "logs/"

LAMBDA = 0.1
EPSILON = 1e-6

NODE_TYPES = {
    "Herb": "Herb",
    "Compound": "inchikey",
    "Target": "Gene_name",
    "GO": "GO_name",
    "Disease": "Disease_name"
}
"""
Node types: dict
    Keys: five types of node
    Values: original labels of in the **_info_all.csv files
"""

NODE_ID = {
    "Herb": "Herb_id",
    "Compound": "Compound_id",
    "Target": "Target_id",
    "GO": "GO_id",
    "Disease": "DisGENet_id"
}
"""
Node id:
    Keys: five types of node
    Values: five column names of node_id
"""

NODE_FILES = {
    "Herb": DATA_PATH+"1.nodes/Herb_info.csv",
    "Compound": DATA_PATH+"1.nodes/Compound_info.csv",
    "Target": DATA_PATH+"1.nodes/Target_info.csv",
    "GO": DATA_PATH+"1.nodes/GO_info.csv",
    "Disease": DATA_PATH+"1.nodes/Disease_info.csv"
}
"""
Node info files:
    Keys: five classes of nodes
    Values: path to five **_info_all.csv files, containing the node_id and original label of each node
"""

EDGE_TYPES = {
    "Herb_Compound": ["Herb", "Compound"],
    "Compound_Target": ["Compound", "Target"],
    "Target_Target": ["Target"],
    "Target_GO": ["Target", "GO"],
    "GO_GO": ["GO"],
    "Target_Disease": ["Target", "Disease"]
}
"""
Edge types: 
    Keys: six types of edge
    Values: endpoint types of each type of edge
"""

EDGE_FILES = {
    "Herb_Compound": DATA_PATH+"2.edges/Herb-Compound.csv",
    "Compound_Target": DATA_PATH+"2.edges/Compound-Protein.csv",
    "Target_Target": DATA_PATH+"2.edges/Target-Target.csv",
    "Target_GO": DATA_PATH+"2.edges/Protein-GO.csv",
    "GO_GO": DATA_PATH+"2.edges/GO-GO.csv",
    "Target_Disease": DATA_PATH+"2.edges/Protein-Disease.csv"
}
"""edge_files:
        keys: six types of edge
        values: path to files of edge info csv
"""

NODE_WEIGHTS = {
    'Compound': 1.0,
    'Target': 1.0,
    'Disease': 1.0,
    'GO': 1.0,
}
"""
Weight of each type of node
"""

POS_PAIR_FILES = {
    "D2C": DATA_PATH+"3.positive_pairs/Positive_Disease_Compound_pair_in_HERB.csv",
    "D2H": DATA_PATH+"3.positive_pairs/Positive_Disease_Herb_pair_in_HERB.csv"
}

PPI_DIST_MAT_FILE = DATA_PATH+"ppi_dist_matrix_uint8.npy"
PPI_DIST_MAT_NODE2ID_FILE = DATA_PATH+"ppi_node_to_id.pkl"

OPTIMAL_PMAP = OUT_PATH + 'pmaps/propagation_map_C1.00_T1.00_D1.00_G1.00_L0.01.pkl'
NETWORK_FP = ROOT_PATH + 'network/net.pkl'