FEATURE_NAMES = [
    "radius1", "texture1", "perimeter1", "area1", "smoothness1", "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2", "smoothness2", "compactness2", "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3"
]
LABEL_NAME = "Diagnosis"
ID_NAME = "ID"
# Column names for the breast cancer dataset
COLUMN_NAMES = [ID_NAME, LABEL_NAME] + FEATURE_NAMES

DATA_FILE_BREAST_CANCER = "data\\breast+cancer+wisconsin+diagnostic\\wdbc.data"
SEED = 42