int initialize(float beam, int max_active, int min_active,
                float acoustic_scale, int allow_partial,
                char* fst_in_filename, char* words_in_filename);
int decode(THFloatTensor *loglikes, THIntTensor *words, THIntTensor *alignments);
