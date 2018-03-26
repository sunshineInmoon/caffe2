#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/interp_op.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2{

template <typename Dtype>
__global__ void interp2_kernel(const int n, const float rheight, const float rwidth,
    const int channels,
    const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      	const int h1 = h2;
      	const int w1 = w2;
	 	const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	  	Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	  	for (int c = 0; c < channels; ++c) {
			pos2[0] = pos1[0];
			pos1 += Width1 * Height1;
			pos2 += Width2 * Height2;
		}
      return;
    }
    //
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    //
    const float w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Dtype w1lambda = w1r - w1;
    const Dtype w0lambda = Dtype(1.) - w1lambda;
    const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
    Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
    for (int c = 0; c < channels; ++c) {
		pos2[0] =
	  	h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) + 
	  	h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
		pos1 += Width1 * Height1;
		pos2 += Width2 * Height2;
    }
  }
}	

template<>
void InterpOp<float,CUDAContext>::interp2(const int channels,
    		const float *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
          	float *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2){

	CAFFE_ENFORCE(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0,"interp.cu file interp2 first has errors!");
  	CAFFE_ENFORCE(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2,"interp.cu file interp2 second has errors!");
	const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  	const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  	const int num_kernels = height2 * width2;
  	interp2_kernel<float><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
    	(num_kernels, rheight, rwidth, channels,
    	data1, x1, y1, height1, width1, Height1, Width1,
     	data2, x2, y2, height2, width2, Height2, Width2);
}

template<>
bool InterpOp<float, CUDAContext>::RunOnDevice(){
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
			
	const auto* xData = X.data<float>();
	auto* yData = Y->mutable_data<float>();
	interp2(num_ * channels_,
    	xData, - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    	yData, 0, 0, height_out_, width_out_, height_out_, width_out_);
			
	return true;
}

template <typename Dtype>
__global__ void interp2_kernel_backward(const int n, const float rheight, const float rwidth,const int channels,
    Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    const Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  	int index = threadIdx.x + blockIdx.x * blockDim.x;
  	if (index < n) {
    	const int w2 = index % width2; // 0:width2-1
    	const int h2 = index / width2; // 0:height2-1
    	// special case: just copy
    if (height1 == height2 && width1 == width2) {
      	const int h1 = h2;
      	const int w1 = w2;
		Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
		const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
		for (int c = 0; c < channels; ++c) {
	  		pos1[0] += pos2[0];
	  		pos1 += Width1 * Height1;
	  		pos2 += Width2 * Height2;
		}
      return;
    }
    //
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    //
    const float w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Dtype w1lambda = w1r - w1;
    const Dtype w0lambda = Dtype(1.) - w1lambda;

    Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
    const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
    for (int c = 0; c < channels; ++c) {
		atomicAdd(&pos1[0], h0lambda * w0lambda * pos2[0]);
		atomicAdd(&pos1[w1p], h0lambda * w1lambda * pos2[0]);
		atomicAdd(&pos1[h1p * Width1], h1lambda * w0lambda * pos2[0]);
		atomicAdd(&pos1[h1p * Width1 + w1p], h1lambda * w1lambda * pos2[0]);
		pos1 += Width1 * Height1;
		pos2 += Width2 * Height2;
    }
  }
}

template<>
void InterpGradientOp<float,CUDAContext>::interp2_backward(const int channels,
	  		float *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    		const float *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2){
	CAFFE_ENFORCE(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0,"interp_op.cu file InterpGradientOp::interp2_backward first check has error!");
  	CAFFE_ENFORCE(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2,"interp_op.cu file InterpGradientOp::interp2_backward second check has error!");
  	const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  	const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  	const int num_kernels = height2 * width2;
  	interp2_kernel_backward<float><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
    	(num_kernels, rheight, rwidth, channels,
     	data1, x1, y1, height1, width1, Height1, Width1,
     	data2, x2, y2, height2, width2, Height2, Width2);
}

template<>
bool InterpGradientOp<float,CUDAContext>::RunOnDevice(){
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
				
	auto* dXData = dX->mutable_data<float>();
	const auto* dYData = dY.data<float>();
	interp2_backward(num_*channels_, dXData,-pad_beg_,-pad_beg_,height_in_eff_,width_in_eff_,height_in_,width_in_,dYData,0,0,height_out_,width_out_,height_out_,width_out_);
	return true;
}

REGISTER_CUDA_OPERATOR(Interp,InterpOp<float,CUDAContext>);
REGISTER_CUDA_OPERATOR(InterpGradient,InterpGradientOp<float,CUDAContext>);
}//namespace caffe2	
