import numpy as np

class Ensemble:

    def __init__(self, texts, sentiment):

        # this should be every text example that will be predicted on
        self._texts = []
        for text in texts:
            self._texts.append(" ".join(str(text).split()))
        # along with its sentiment
        self._sentiment = sentiment

        # initialize accumulators: these lists will be of len num_models * num_folds
        self._preds_start = []
        self._preds_end = []
        self._offsets = []
        self._weights = []


    def add(self, preds_start, preds_end, offsets, weight=1, byte_level=False):
        """

        This function will be called, to add predictions and offsets per fold per model
        (if not clear see example below)

        Note: it's important that preds_start and preds_end has been softmax'ed

        Note: the weight determines how much weight we should give this batch of
        predictions. e.g. if this is model_1_fold_X, and we know model_1 is relatively
        weak, we can give it a weight of e.g. 0.5 (instead of default 1)
        """

        # because self._texts won't have a preceeding blankspace we need
        # to subtract ByteLevel tokenizers offsets by one (except when offset = 0)
        if byte_level:
            corrected_offsets = []
            for offset in offsets:
                _offset = []
                for o1, o2 in offset:
                    _offset.append((o1-1 if o1 != 0 else o1, o2-1 if o2 != 0 else o2))
                corrected_offsets.append(_offset)
            offsets = np.asarray(corrected_offsets, dtype=np.int32)

        self._preds_start.append(preds_start)
        self._preds_end.append(preds_end)
        self._weights.append(weight)
        self._offsets.append(offsets)


    def _token_logits_to_char_logits(self):
        """
        *** This function will not be called directly. ***

        first loops over folds of all models, then over each example:

            for _, (_, _, _) in [model_1_fold_1, model_1_fold_2, ..., model_N_fold_5]:
                for _, _, (_, _) in [example_1, example_2, ..., example_N]:
                    char_preds[i, j:k] += logits

        """

        # initialize char preds arrays for later
        self.char_preds_start = [
            np.zeros(len(text), dtype=np.float32) for text in self._texts
        ]
        self.char_preds_end = [
            np.zeros(len(text), dtype=np.float32) for text in self._texts
        ]

        # loop over folds * models
        for preds_start, preds_end, weight, offsets in zip(
            self._preds_start, self._preds_end, self._weights, self._offsets):

            # loop over each example (or ID)
            for i, (pred_start, pred_end, offset) in enumerate(zip(preds_start, preds_end, offsets)):
                for pstart, pend, (o1, o2) in zip(pred_start, pred_end, offset):
                    self.char_preds_start[i][o1:o2] += pstart * weight
                    self.char_preds_end[i][o1:o2] += pend * weight


    def _char_logits_to_word_logits(self):
        """
        *** This function will not be called directly. ***

        loops over each example, and looks for the index of maximum value in
        both char_pred_start and char_pred_end, then uses these positions
        for final prediction.

        Note: argmax is customized.

        """

        def argmax(a, take='first'):
            if take == 'first':
                return np.where(a == a.max())[0][0]
            elif take == 'last':
                return np.where(a == a.max())[0][-1]

        self.selected_text_preds = []
        for text, sentiment, char_pred_start, char_pred_end in zip(
            self._texts, self._sentiment, self.char_preds_start, self.char_preds_end):

            pos_start = argmax(char_pred_start, take='first')
            pos_end = argmax(char_pred_end, take='last')

            if sentiment == "neutral" or len(text.split()) < 2:
                text_pred = text
            elif pos_start > pos_end: # pos_start > pos_end rarely happens, but we should have this clause still.
                text_pred = ""
                for i in range(len(text)):
                    if i == pos_start:
                        # "or i == pos_start" because ByteLeveltokenizer will sometimes
                        # have a blankspace before the word, for example:
                        # text[pos_start] == " "; text[pos_start:pos_end+1] == " love"
                        while text[i] != " " or i == pos_start:
                            text_pred += text[i]
                            i += 1
                            # we need to break the while loop if index
                            # is equal or greater than text len, otherwise
                            # it will raise an error ("index out of range")
                            if i >= len(text):
                                break
            else:
                text_pred = text[pos_start:pos_end+1]

            self.selected_text_preds.append(text_pred)

    def compute_predictions(self):
        """This function will be called to obtain selected_text_preds for the submission file"""
        self._token_logits_to_char_logits()
        self._char_logits_to_word_logits()
        return self.selected_text_preds
