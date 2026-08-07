from preprocessing.preprocess_pipeline import PreprocessPipeline

pipeline = PreprocessPipeline(
    clip_length=16
)

pipeline.process_dataset(
    "data/raw"
)