class DetectionMetrics:

    @staticmethod
    def precision(tp, fp):

        return tp/(tp+fp) if tp+fp else 0

    @staticmethod
    def recall(tp, fn):

        return tp/(tp+fn) if tp+fn else 0

    @staticmethod
    def f1(p, r):

        return 2*p*r/(p+r) if p+r else 0
