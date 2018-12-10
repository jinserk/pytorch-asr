import torch
from torch.nn.modules.loss import _Loss

from asr.utils.misc import onehot2int

class EditDistanceLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target, input_seq_lens, target_seq_lens):
        """
        input: BxTxH, target: BxN, input_seq_lens: B, target_seq_lens: B
        """
        batch_size = input.size(0)
        eds = list()
        for b in range(batch_size):
            x = torch.argmax(input[b, :input_seq_lens[b]], dim=-1)
            y = target[b, :target_seq_lens[b]]
            d = self.calculate_levenshtein(x, y)
            eds.append(d)
        loss = torch.FloatTensor(eds)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()

    def calculate_levenshtein(self, seq1, seq2):
        """
        implement the extension of the Wagner–Fischer dynamic programming algorithm
        """
        size_x, size_y = len(seq1), len(seq2)
        matrix = torch.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                cost = 0 if seq1[x] == seq2[y] else 1
                comps = torch.LongTensor([
                    matrix[x - 1, y] + 1,               # deletion
                    matrix[x, y - 1] + 1,               # insertion
                    matrix[x - 1, y - 1] + cost,        # subtitution
                ])
                matrix[x, y] = torch.min(comps)
                if x > 1 and y > 1 and seq1[x] == seq2[y - 1] and seq1[x - 1] == seq2[y]:
                    comps = torch.LongTensor([
                        matrix[x, y],
                        matrix[x - 2, y - 2] + cost,    # transposition
                    ])
                    matrix[x, y] = torch.min(comps)

        return matrix[-1, -1]

if __name__ == "__main__":
    x = torch.LongTensor([0, 1, 2])
    y = torch.LongTensor([0, 2, 1, 3])
    l = EditDistanceLoss()
    print(l.calculate_levenshtein(x, y))
