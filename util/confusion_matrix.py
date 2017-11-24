import numpy as np
class ConfusionMatrix:
    def __init__(self, size):
        self.size = size
        self.diag = np.zeros(self.size)
        self.act_sum = np.zeros(self.size)
        self.pre_sum = np.zeros(self.size)

    def reset(self):
        self.diag = np.zeros(self.size)
        self.act_sum = np.zeros(self.size)
        self.pre_sum = np.zeros(self.size)

    def update(self, actual, predicted):
        for i in range(self.size):
            act = actual == i
            pre = predicted == i
            I = act & pre
            self.diag[i] += np.sum(I)
            self.act_sum[i] += np.sum(act)
            self.pre_sum[i] += np.sum(pre)

    def accuracy(self):
        ''' accuracy '''
        diag_sum = np.sum(self.diag)
        total_sum = np.sum(self.act_sum)
        if total_sum == 0:
            return 0
        else:
            return diag_sum / total_sum

    def fg_accuracy(self):
        '''fg_accuracy'''
        diag_sum = np.sum(self.diag) - self.diag[0]
        total_sum = np.sum(self.act_sum) - self.act_sum[0]
        if total_sum == 0:
            return 0
        else:
            return diag_sum / total_sum

    def avg_precision(self):
        '''avg_precision: ignore the label that isn't in imgs of gt'''
        total_precision = 0
        count = 0
        for i in range(self.size):
            if self.pre_sum[i] > 0:
                total_precision += self.diag[i] / self.pre_sum[i]
                count += 1
        if count == 0:
            return 0
        else:
            return total_precision / count

    def avg_recall(self):
        '''avg_recall: ignore the label that isn't in imgs of gt'''
        total_recall = 0
        count = 0
        for i in range(self.size):
            if self.act_sum[i] > 0:
                total_recall += self.diag[i] / self.act_sum[i]
                count += 1
        if count == 0:
            return 0
        else:
            return total_recall / count

    def avg_f1score(self):
        '''avgF1score: ignore the label that isn't in imgs of gt'''
        total_f1score = 0
        count = 0
        for i in range(self.size):
            t = self.pre_sum[i] + self.act_sum[i]
            if t > 0:
                total_f1score += 2 * self.diag[i] / t
                count += 1
        if count == 0:
            return 0
        else:
            return total_f1score / count

    def f1score(self):
        '''F1score: ignore the label that isn't in imgs of gt'''
        f1score = []
        for i in range(self.size):
            t = self.pre_sum[i] + self.act_sum[i]
            if t > 0:
                f1score.append(2 * self.diag[i] / t)
            else:
                f1score.append(-1)
        return f1score

    def print_f1score(self):
        '''F1score: ignore the label that isn't in imgs of gt'''
        f1score = ''
        for i in range(self.size):
            t = self.pre_sum[i] + self.act_sum[i]
            if t > 0:
                f1score += ((str(2 * self.diag[i] / t)) + ' ')
            else:
                f1score += '-1 '
        return f1score

    def mean_iou(self):
        '''meanIoU: ignore the label that isn't in imgs of gt'''
        total_iou = 0
        count = 0
        for i in range(self.size):
            I = self.diag[i]
            U = self.act_sum[i] + self.pre_sum[i] - I
            if U > 0:
                total_iou += I / U
                count += 1
        return total_iou / count