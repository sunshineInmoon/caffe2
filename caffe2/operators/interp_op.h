#ifndef CAFFE2_OPERATORS_INTER_OP_H
#define CAFFE2_OPERATORS_INTER_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class InterpOp final : public Operator<Context>{
	public:
		USE_OPERATOR_CONTEXT_FUNCTIONS;
		InterpOp(const OperatorDef& def, Workspace* ws)
			:Operator<Context>(def,ws),
			zoom_factor_(OperatorBase::template GetSingleArgument<int>("zoom_factor",0)),
			shrink_factor_(OperatorBase::template GetSingleArgument<int>("shrink_factor",0)),
			height_(OperatorBase::template GetSingleArgument<int>("height",0)),
			width_(OperatorBase::template GetSingleArgument<int>("width",0)),
			pad_beg_(OperatorBase::template GetSingleArgument<int>("pad_beg",0)),
			pad_end_(OperatorBase::template GetSingleArgument<int>("pad_end",0)){
				int num_specs = 0;
				CAFFE_ENFORCE_LE(def.input_size(), 2,"bottom number of interp layer should be 1 or 2 ");
  				num_specs += zoom_factor_;
  				num_specs += shrink_factor_;
 				num_specs += height_ && width_;
				num_specs += (def.input_size() == 2);
  				CAFFE_ENFORCE_EQ(num_specs, 1, "Output dimension specified either by zoom factor or shrink factor or explicitly");
  				CAFFE_ENFORCE_LE(pad_beg_, 0 ,"Only supports non-pos padding (cropping) for now");
  				CAFFE_ENFORCE_LE(pad_end_, 0 ,"Only supports non-pos padding (cropping) for now");
			}
		bool RunOnDevice() override;
	private:
		int num_, channels_;
  		int height_in_, width_in_;
  		int height_out_, width_out_;
 		int pad_beg_, pad_end_;
 		int height_in_eff_, width_in_eff_;
		
		int zoom_factor_;
		int shrink_factor_;
		int height_,width_;

		Tensor<Context> X_1;
		
		void interp2(const int channels,
    		const T *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
          	T *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);

};

template <typename T, class Context>
class InterpGradientOp final : public Operator<Context>{
	public:
		USE_OPERATOR_CONTEXT_FUNCTIONS;
		InterpGradientOp(const OperatorDef& def, Workspace* ws)
			:Operator<Context>(def, ws),
			zoom_factor_(OperatorBase::template GetSingleArgument<int>("zoom_factor",0)),
			shrink_factor_(OperatorBase::template GetSingleArgument<int>("shrink_factor",0)),
			height_(OperatorBase::template GetSingleArgument<int>("height",0)),
			width_(OperatorBase::template GetSingleArgument<int>("width",0)),
			pad_beg_(OperatorBase::template GetSingleArgument<int>("pad_beg",0)),
			pad_end_(OperatorBase::template GetSingleArgument<int>("pad_end",0)){
				int num_specs = 0;
				CAFFE_ENFORCE_LE(def.input_size(), 2,"bottom number of interp layer should be 1 or 2 ");
  				num_specs += zoom_factor_;
  				num_specs += shrink_factor_;
 				num_specs += height_ && width_;
				num_specs += (def.input_size() == 2);
  				CAFFE_ENFORCE_EQ(num_specs, 1, "Output dimension specified either by zoom factor or shrink factor or explicitly");
  				CAFFE_ENFORCE_LE(pad_beg_, 0 ,"Only supports non-pos padding (cropping) for now");
  				CAFFE_ENFORCE_LE(pad_end_, 0 ,"Only supports non-pos padding (cropping) for now");
			}
			
			bool RunOnDevice() override;
			
	private:
		int num_, channels_;
  		int height_in_, width_in_;
  		int height_out_, width_out_;
 		int pad_beg_, pad_end_;
 		int height_in_eff_, width_in_eff_;
		
		int zoom_factor_;
		int shrink_factor_;
		int height_,width_;
		
		Tensor<Context> X_1;

		
		void interp2_backward(const int channels,
	  		T *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    		const T *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);
};

}

#endif
