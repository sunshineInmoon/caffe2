#include "caffe2/operators/interp_op.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <typename T, class Context>
void InterpOp<T,Context>::interp2(const int channels,
    		const T *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
          	T *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2){
	CAFFE_ENFORCE(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0,"caffe_cpu_interp2() first check has errors!");
  CAFFE_ENFORCE(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2,"caffe_cpu_interp2() second check has errors!");
  // special case: just copy
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
	  	const int w1 = w2;
	 	 const T* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	  	T* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	 	 for (int c = 0; c < channels; ++c) {
	   	 pos2[0] = pos1[0];
	    	pos1 += Width1 * Height1;
	    	pos2 += Width2 * Height2;
	 	 }
      }
    }
    return;
  }
  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const T h1lambda = h1r - h1;
    const T h0lambda = T(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const T w1lambda = w1r - w1;
      const T w0lambda = T(1.) - w1lambda;
	  const T* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
		T* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
		for (int c = 0; c < channels; ++c) {
	  		pos2[0] =
	    		h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) + 
	    		h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
	  		pos1 += Width1 * Height1;
	 		pos2 += Width2 * Height2;
		}
    }
  }
}	

template<>
bool InterpOp<float,CPUContext>::RunOnDevice(){
	const auto& X = Input(0);
	if(OperatorBase::InputSize()==2){
		X_1.CopyFrom(Input(1));
	}
	num_ = X.dim32(0);
	channels_ = X.dim32(1);
	height_in_ = X.dim32(2);
	width_in_ = X.dim32(3);
			
	height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
  	width_in_eff_ = width_in_ + pad_beg_ + pad_end_;
			
	if (zoom_factor_ != 0) {
    	CAFFE_ENFORCE_GE(zoom_factor_, 1, "Zoom factor must be positive");
    	height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor_ - 1);
    	width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor_ - 1);
 	}
  	else if (shrink_factor_ != 0) {
  		CAFFE_ENFORCE_GE(shrink_factor_, 1, "Shrink factor must be positive");
   		height_out_ = (height_in_eff_ - 1) / shrink_factor_ + 1;
    	width_out_ = (width_in_eff_ - 1) / shrink_factor_ + 1;
  	}
  	else if ((height_!=0) && (width_!=0)) {
    	height_out_  = height_;
    	width_out_  = width_;
  	}
	else if (OperatorBase::InputSize() == 2) {
    	height_out_  = X_1.dim32(2);
    	width_out_  = X_1.dim32(3);
  	}
  	else {
    	LOG(FATAL); 
  	}
  	CAFFE_ENFORCE_GT(height_in_eff_, 0, "height should be positive");
  	CAFFE_ENFORCE_GT(width_in_eff_, 0, "width should be positive");
  	CAFFE_ENFORCE_GT(height_out_, 0, "height should be positive");
  	CAFFE_ENFORCE_GT(width_out_, 0, "width should be positive");
			
	vector<int> Dims;
	Dims.push_back(num_);
	Dims.push_back(channels_);
	Dims.push_back(height_out_);
	Dims.push_back(width_out_);
	auto* Y = Output(0);
	Y->Reshape(Dims);
			
	const auto* xData = X.template data<float>();
	auto* yData = Y->template mutable_data<float>();
	interp2(num_ * channels_,
    	xData, - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    	yData, 0, 0, height_out_, width_out_, height_out_, width_out_);

	return true;
}

	
template <typename T, class Context>
void InterpGradientOp<T,Context>::interp2_backward(const int channels,
	  		T *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    		const T *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2){
  CAFFE_ENFORCE(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0,"caffe_cpu_interp2_backward() first check has errors!");
  CAFFE_ENFORCE(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2,"caffe_cpu_interp2_backward() second check has errors!");
  // special case: same-size matching grids
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
		const int w1 = w2;
	  	T* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	  	const T* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	  	for (int c = 0; c < channels; ++c) {
	    	pos1[0] += pos2[0];//注意前向是pos2[0]+=pos1[0]
	    	pos1 += Width1 * Height1;
	   		pos2 += Width2 * Height2;
	  	}
      }
    }
    return;
  }
  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const T h1lambda = h1r - h1;
    const T h0lambda = T(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
    	const float w1r = rwidth * w2;
      	const int w1 = w1r;
      	const int w1p = (w1 < width1 - 1) ? 1 : 0;
      	const T w1lambda = w1r - w1;
      	const T w0lambda = T(1.) - w1lambda;
	  	T* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	  	const T* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
		for (int c = 0; c < channels; ++c) {
	 		pos1[0] += h0lambda * w0lambda * pos2[0];
	  		pos1[w1p] += h0lambda * w1lambda * pos2[0];
	  		pos1[h1p * Width1] += h1lambda * w0lambda * pos2[0];
	  		pos1[h1p * Width1 + w1p] += h1lambda * w1lambda * pos2[0];
	  		pos1 += Width1 * Height1;
	  		pos2 += Width2 * Height2;
		}
    }
  }
}

template<>
bool InterpGradientOp<float,CPUContext>::RunOnDevice(){
	const auto& dY = Input(0);
	const auto& X = Input(1);
	if(OperatorBase::InputSize()==3){
		X_1.CopyFrom(Input(2));
	}
	auto* dX = Output(0);
	dX->ResizeLike(X);
				
	num_ = X.dim32(0);
	channels_ = X.dim32(1);
	height_in_ = X.dim32(2);
	width_in_ = X.dim32(3);
			
	height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
  	width_in_eff_ = width_in_ + pad_beg_ + pad_end_;
				
	if (zoom_factor_ != 0) {
    	CAFFE_ENFORCE_GE(zoom_factor_, 1, "Zoom factor must be positive");
    	height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor_- 1);
    	width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor_ - 1);
 	}
  	else if (shrink_factor_ != 0) {
  		CAFFE_ENFORCE_GE(shrink_factor_, 1, "Shrink factor must be positive");
   		height_out_ = (height_in_eff_ - 1) / shrink_factor_ + 1;
    	width_out_ = (width_in_eff_ - 1) / shrink_factor_ + 1;
  	}
  	else if ((height_!=0) && (width_!=0)) {
    	height_out_  = height_;
    	width_out_  = width_;
  	}
	else if (OperatorBase::InputSize() == 3) {
    	height_out_  = X_1.dim32(2);
    	width_out_  = X_1.dim32(3);
  	}
  	else {
    	LOG(FATAL); 
  	}
  	CAFFE_ENFORCE_GT(height_in_eff_, 0, "height should be positive");
  	CAFFE_ENFORCE_GT(width_in_eff_, 0, "width should be positive");
  	CAFFE_ENFORCE_GT(height_out_, 0, "height should be positive");
  	CAFFE_ENFORCE_GT(width_out_, 0, "width should be positive");
				
	auto* dXData = dX->template mutable_data<float>();
	const auto* dYData = dY.template data<float>();
	interp2_backward(num_*channels_, dXData,
         -pad_beg_,-pad_beg_,height_in_eff_,width_in_eff_,height_in_,width_in_,dYData,
         0,0,height_out_,width_out_,height_out_,width_out_);
	return true;
}

REGISTER_CPU_OPERATOR(Interp, InterpOp<float,CPUContext>);
OPERATOR_SCHEMA(Interp)
	.NumInputs(1,2)
	.NumOutputs(1)
	.Arg("zoom_factor","zoom a image")
	.Arg("shrink_factor","shrink a image")
	.Arg("height","the height of out image")
	.Arg("width","the width of out image")
	.Arg("pad_beg","pad begin ")
	.Arg("pad_end","pad end")
	.SetDoc(R"DOC(Bilinear Interpolation)DOC");
	
REGISTER_CPU_OPERATOR(InterpGradient,InterpGradientOp<float,CPUContext>);
OPERATOR_SCHEMA(InterpGradient)
	.NumInputs(2,3)
	.NumInputs(1)
	.Arg("zoom_factor","zoom a image")
	.Arg("shrink_factor","shrink a image")
	.Arg("height","the height of out image")
	.Arg("width","the width of out image")
	.Arg("pad_beg","pad begin ")
	.Arg("pad_end","pad end")
	.SetDoc(R"DOC(Bilinear Interpolation Gradient)DOC");

class GetInterpGradient : public GradientMakerBase{
	using GradientMakerBase::GradientMakerBase;
	vector<OperatorDef> GetGradientDefs() override{
		return SingleGradientDef(
			"InterpGradient",
			"",
			std::vector<string>{GO(0),I(0),I(1)},
			std::vector<string>{GI(0)});
	}
};
	
}
