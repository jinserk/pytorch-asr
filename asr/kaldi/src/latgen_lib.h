int 
initialize(float beam, int max_active, int min_active,
           float acoustic_scale, int allow_partial,
           char* fst_in_filename, char* words_in_filename);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
decode(torch::Tensor loglikes, torch::Tensor frame_lens);
