from evaluation.detection.evaluator import DetectionEvaluator

predictions = [
    # Detection objects
]

ground_truth = [
    [20,30,140,170],
    [200,50,320,180]
]

evaluator = DetectionEvaluator()

result = evaluator.evaluate(
    predictions,
    ground_truth
)

print(result)
