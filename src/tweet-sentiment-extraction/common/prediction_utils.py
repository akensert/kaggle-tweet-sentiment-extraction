import numpy as np

def transform_to_text(pred_start, pred_end, text, offset, sentiment):

    def decode(pred_start, pred_end, text, offset):
        decoded_text = ""
        for i in range(pred_start, pred_end+1):
            decoded_text += text[offset[i][0]:offset[i][1]]
            if (i+1) < len(offset) and offset[i][1] < offset[i+1][0]:
                decoded_text += " "
        return decoded_text

    decoded_predictions = []
    for i in range(len(text)):
        # if sentiment[i] == "neutral" or len(text[i].split()) < 2:
        #     decoded_text = text[i]
        # else:
        idx_start = np.argmax(pred_start[i])
        # idx_end = np.argmax(pred_end[i])
        candidates_end = np.argsort(pred_end[i])[::-1]

        j = 0
        while 1:
            idx_end = candidates_end[j]
            if idx_start <= idx_end:
                break
            j += 1

        decoded_text = str(decode(idx_start, idx_end, text[i], offset[i]))
        # if len(decoded_text) == 0:
        #     decoded_text = text[i]
        decoded_predictions.append(decoded_text)

    return decoded_predictions


def compute_jaccard(selected_text, selected_text_pred):

    def jaccard(str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    jaccard_mean = 0.
    for i in range(len(selected_text)):
        jaccard_mean += jaccard(selected_text[i], selected_text_pred[i])
    return jaccard_mean / len(selected_text)
