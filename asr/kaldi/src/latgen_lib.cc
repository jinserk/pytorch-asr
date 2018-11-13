// latgen.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

// modified by Jinserk Baik <jinserk.baik@gmail.com>

#include <tuple>
#include <sstream>

#include <torch/extension.h>

#include "base/kaldi-common.h"
#include "util/kaldi-thread.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc


using namespace kaldi;

typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;
using fst::ReadFstKaldi;


struct LatticeDecoderOptions
{
	FasterDecoderOptions decoder_opts_;

	BaseFloat acoustic_scale_;
	bool allow_partial_;

	VectorFst<StdArc>* decode_fst_ = NULL;
	fst::SymbolTable* word_syms_ = NULL;

	LatticeDecoderOptions()
	: acoustic_scale_(1.0),
	  allow_partial_(true)
	{}

	~LatticeDecoderOptions()
	{
		delete decode_fst_;
		delete word_syms_;
	}

	void update_decoder_options(BaseFloat beam, int32 max_active, int32 min_active,
								BaseFloat beam_delta = 0.5, BaseFloat hash_ratio = 2.0)
	{
		decoder_opts_.beam = beam;
		decoder_opts_.max_active = max_active;
		decoder_opts_.min_active = min_active;
		decoder_opts_.beam_delta = beam_delta;
		decoder_opts_.hash_ratio = hash_ratio;
	}

	void load_files(std::string fst_in_filename, std::string words_in_filename)
	{
		if (decode_fst_) delete decode_fst_;
		decode_fst_ = fst::ReadFstKaldi(fst_in_filename);
		if (!decode_fst_)
			KALDI_ERR << "Could not read decoding graph from file " << fst_in_filename;

		if (word_syms_) delete word_syms_;
		word_syms_ = fst::SymbolTable::ReadText(words_in_filename);
		if (!word_syms_)
			KALDI_ERR << "Could not read symbol table from file " << words_in_filename;
	}

}; // struct LatticeDecoderOptions


struct LatticeDecoderResult
{
	std::vector<int32> alignments_;
	std::vector<int32> words_;
	//std::string text_;
	bool failed_ = false;
	bool partial_ = false;

}; // struct LatticeDecoderResult


class LatticeDecoder
{
	private:
		LatticeDecoderOptions& opts_;
		FasterDecoder decoder_;

	public:
		LatticeDecoder(LatticeDecoderOptions& opts)
		: opts_(opts),
		  decoder_(*opts_.decode_fst_, opts_.decoder_opts_)
		{}

		int decode(Matrix<BaseFloat>& loglikes, LatticeDecoderResult& result)
		{
			int num_fail = 0;

			if (loglikes.NumRows() == 0) {
				num_fail++;
				result.failed_ = true;
				return num_fail;
			}

			DecodableMatrixScaled decodable(loglikes, opts_.acoustic_scale_);
			decoder_.Decode(&decodable);

			VectorFst<LatticeArc> decoded;  // linear FST.

			if ((opts_.allow_partial_ || decoder_.ReachedFinal())
				&& decoder_.GetBestPath(&decoded)) {
				result.partial_ = !decoder_.ReachedFinal();
				LatticeWeight weight;
				GetLinearSymbolSequence(decoded, &result.alignments_, &result.words_, &weight);
				/*
				std::stringstream ss;
				for (auto w : result.words_)
					ss << opts_.word_syms_->Find(w) << ' ';
				result.text_ = ss.str().substr(0, ss.str().length()-1);
				*/
			}
			else {
				num_fail++;
				result.failed_ = true;
			}

			return num_fail;
		}

}; // class LatticeDecoder


class DecodingTask
{

	private:
		LatticeDecoder decoder_;

	public:
		Matrix<BaseFloat>& loglikes_;
		LatticeDecoderResult& result_;

		DecodingTask (LatticeDecoderOptions& opts,
					  Matrix<BaseFloat>& loglikes,
					  LatticeDecoderResult& result)
		: decoder_(opts),
		  loglikes_(loglikes),
		  result_(result)
		{ }

		void operator() ()
		{
			decoder_.decode(loglikes_, result_);
		}

}; // class SttSpeakerTask


// global settings
#define DEFAULT_NJOBS   3

TaskSequencerConfig sequencer_config;
LatticeDecoderOptions latgen_opts;


int
initialize(float beam, int max_active, int min_active,
           float acoustic_scale, int allow_partial,
           char* fst_in_filename, char* words_in_filename)
{
	ParseOptions po ("");
	sequencer_config.Register (&po);

	// set concurrent number of jobs
	int njobs = DEFAULT_NJOBS;
	int n = std::thread::hardware_concurrency () - 1;
	if (n < njobs)
		g_num_threads = sequencer_config.num_threads = n;
	else
		g_num_threads = sequencer_config.num_threads = njobs;

	sequencer_config.num_threads = g_num_threads;
	sequencer_config.num_threads_total = g_num_threads;

	latgen_opts.acoustic_scale_ = acoustic_scale;
	latgen_opts.allow_partial_ = allow_partial;

	latgen_opts.update_decoder_options(beam, max_active, min_active);
	latgen_opts.load_files(fst_in_filename, words_in_filename);

	return 1;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
decode(torch::Tensor loglikes, torch::Tensor frame_lens)
{
	const auto num_batch = loglikes.size(0);
	const auto num_frame = loglikes.size(1);
	const auto num_class = loglikes.size(2);

	// convert torch::Tensor to list of kaldi::SubMatrix
	std::vector<Matrix<BaseFloat> > loglikes_list;
	std::vector<LatticeDecoderResult> results;
	for (int b = 0; b < num_batch; b++) {
		loglikes_list.emplace_back(SubMatrix<BaseFloat>((float*)loglikes[b].data_ptr(), frame_lens[b].item<int>(), num_class, num_class));
		results.emplace_back(LatticeDecoderResult());
	}

	/*
	// decode
	LatticeDecoder decoder(latgen_opts);
	decoder.decode(loglikes_list, results);
	*/

	TaskSequencer<DecodingTask> task_sequencer(sequencer_config);
	for (int b = 0; b < num_batch; b++) {
		DecodingTask *task = new DecodingTask(latgen_opts, loglikes_list[b], results[b]);
		task_sequencer.Run(task);
	}
	task_sequencer.Wait();


	// get max length
	int max_words = 0, max_alignments = 0;
	for (auto &r : results) {
		if (r.failed_) continue;
		if (max_words < r.words_.size())
			max_words = r.words_.size();
		if (max_alignments < r.alignments_.size())
			max_alignments = r.alignments_.size();
	}

	// prepare output
	const long sz = static_cast<long>(results.size());
	auto opt = loglikes.options().dtype(at::kInt);
	auto words = torch::zeros({sz, max_words}, opt);
	auto alignments = torch::zeros({sz, max_alignments}, opt);
	auto w_sizes = torch::zeros({sz, }, opt);
	auto a_sizes = torch::zeros({sz, }, opt);
	for (int i = 0; i < sz; i++) {
		if (results[i].failed_) continue;
		//strncpy(texts[i], results[i].text_.c_str(), results[i].text_.length());
		w_sizes[i] = (int)results[i].words_.size();
		a_sizes[i] = (int)results[i].alignments_.size();
		for (int j = 0; j < results[i].words_.size(); j++)
			words[i][j] = (int)results[i].words_[j];
		for (int j = 0; j < results[i].alignments_.size(); j++)
			alignments[i][j] = (int)results[i].alignments_[j];
	}

	return std::make_tuple(words, alignments, w_sizes, a_sizes);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("initialize", &initialize, "initialize Kaldi latgen decoder");
  m.def("decode", &decode, "decode with Kaldi latgen decoder");
}

