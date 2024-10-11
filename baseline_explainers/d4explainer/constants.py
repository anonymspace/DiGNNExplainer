feature_dict = {
    "BA_shapes": 10,
    "Tree_Cycle": 10,
    "Tree_Grids": 10,
    "mutag": 7,
    "ba3": 4,
    "dblp": 4,
    "imdb": 3,
    "pubmed": 8
}

task_type = {
    "BA_shapes": "nc",
    "Tree_Cycle": "nc",
    "Tree_Grids": "nc",
    "mutag": "gc",
    "ba3": "gc",
    "dblp": "nc",
    "imdb": "nc",
    "pubmed": "nc"
}

dataset_choices = list(task_type.keys())
