import os
folder = "data/raw/DFD_manipulated_sequences/DFD_manipulated_sequences"
print(len([f for f in os.listdir(folder) if f.lower().endswith(".mp4")]))

# comment out the actual pipeline run for now
# pipeline = PreprocessPipeline(clip_length=16)
# pipeline.process_dataset("data/raw")