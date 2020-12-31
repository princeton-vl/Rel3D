import torch

class Ensemble:

    def __init__(self, model1, model2, lambda_rel):
        """
        :param model1: should be initialized
        :param model2: should be initialized
        """

        self.model1 = model1
        self.model2 = model2
        self.lambda_rel = lambda_rel

    def __call__(self, inp_model1, inp_model2, rels):
        logit1 = self.model1(**inp_model1)
        logit2 = self.model2(**inp_model2)
        prob = [(_logit1 * self.lambda_rel[rel]) + (_logit2 * (1 - self.lambda_rel[rel]))
                for _logit1, _logit2, rel in zip(logit1, logit2, rels)]
        prob = torch.tensor(prob).to(logit1.device, non_blocking=True)
        return prob

    def eval(self):
        self.model1.eval()
        self.model2.eval()