#ifndef TYPEEXG_MATLAB_EIG_H
#define TYPEEXG_MATLAB_EIG_H

// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

#include "Eigen/Dense"
#include "mex.h"
#include <vector>

#define EigenMatrix Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
#define EIGEN_NO_DEBUG // to remove bound checking to get faster element access

// this namespace contains helper functions to be used only in this file (not be called from outside)
// they are all put in a namespace to avoid clashing (resulting in linker errors) with other same
// function names in other header files
namespace hpers_TEMatEig
{
	// Template for mapping C primitive types to MATLAB types (commented out ones are not supported in Eigen)
	template<class T> inline mxClassID getMatlabType() { return mxUNKNOWN_CLASS; }
	template<> inline mxClassID getMatlabType<char>() { return mxINT8_CLASS; }
	template<> inline mxClassID getMatlabType<unsigned char>() { return mxUINT8_CLASS; }
	template<> inline mxClassID getMatlabType<short>() { return mxINT16_CLASS; }
	template<> inline mxClassID getMatlabType<unsigned short>() { return mxUINT16_CLASS; }
	template<> inline mxClassID getMatlabType<int>() { return mxINT32_CLASS; }
	template<> inline mxClassID getMatlabType<unsigned int>() { return mxUINT32_CLASS; }
	template<> inline mxClassID getMatlabType<long long>()			{ return mxINT64_CLASS; }
	template<> inline mxClassID getMatlabType<unsigned long long>() { return mxUINT64_CLASS; }
	template<> inline mxClassID getMatlabType<float>() { return mxSINGLE_CLASS; }
	template<> inline mxClassID getMatlabType<double>() { return mxDOUBLE_CLASS; }	
}


// works for 2D matrices (real numbers, not complex)
// assumes that underlying matrix data is stored in column major order.
template <typename T>
void eigen2matlab(const EigenMatrix& matIn, mxArray* &matOut)
{	
	int nrows = matIn.rows();
	int ncols = matIn.cols();
	matOut = mxCreateNumericMatrix(nrows, ncols, hpers_TEMatEig::getMatlabType<T>(), mxREAL);
	T *dst_pointer = (T*)mxGetData(matOut);
	const T *src_pointer = (T*)matIn.data();
	std::memcpy(dst_pointer, src_pointer, sizeof(T)*nrows*ncols);
}

// works for 3D matrices (real numbers, not complex)
// a 3D matrix can be thought of as a 2D matrix with any number of channels
// assumes that underlying matrix data is stored in column major order.
template <typename T>
void eigen2matlab(const std::vector<EigenMatrix> &matIn, mxArray* &matOut)
{
	int nrows = matIn[0].rows();
	int ncols = matIn[0].cols();
	int nchannels = matIn.size();
	const size_t ndims = 3;
	size_t dims[ndims] = { nrows, ncols, nchannels };
	matOut = mxCreateNumericArray(ndims, dims, hpers_TEMatEig::getMatlabType<T>(), mxREAL);
	T *dst_pointer = (T*)mxGetData(matOut);
	T *tmp_pointer = dst_pointer;

	for (int i = 0; i < nchannels; i++)
	{
		const T *src_pointer = (T*)matIn[i].data();
		std::memcpy(tmp_pointer, src_pointer, sizeof(T)*nrows*ncols);
		tmp_pointer += (nrows * ncols);
	}

	tmp_pointer = NULL;
}


// works for 2D matrices (real numbers, not complex)
// assumes that underlying matrix data is stored in column major order.
template <typename T>
void matlab2eigen(mxArray* matIn, EigenMatrix &matOut, bool copy_mem = false)
{	
	int nrows = mxGetM(matIn);
	int ncols = mxGetN(matIn);

	T *src_pointer = (T*)mxGetData(matIn);	

	if (copy_mem)
	{		
		matOut = EigenMatrix(nrows, ncols);
		T *dst_pointer = (T*)matOut.data();
		std::memcpy(dst_pointer, src_pointer, sizeof(T)*nrows*ncols);
	}
	else
	{
		matOut = Eigen::Map<EigenMatrix>(src_pointer, nrows, ncols);		
	}
		
}

// works for 3D matrices (real numbers, not complex)
// a 3D matrix can be thought of as a 2D matrix with any number of channels
// assumes that underlying matrix data is stored in column major order.
template <typename T>
void matlab2eigen(mxArray* matIn, std::vector<EigenMatrix> &matOut, bool copy_mem = true)
{
	int ndims = (int)mxGetNumberOfDimensions(matIn);
	const size_t *dims = mxGetDimensions(matIn);
	unsigned int nrows = (unsigned int)dims[0];
	unsigned int ncols = (unsigned int)dims[1];
	unsigned int nchannels = ndims == 2 ? 1 : (unsigned int)dims[2];

	T *src_pointer = (T*)mxGetData(matIn);

	matOut.resize(nchannels);

	if (copy_mem)
	{
		for (int i = 0; i < nchannels; i++)
		{
			matOut[i] = EigenMatrix(nrows, ncols);
			T *dst_pointer = (T*)matOut[i].data();
			std::memcpy(dst_pointer, src_pointer, sizeof(T)*nrows*ncols);
			src_pointer += (nrows * ncols);
		}
		
	}
	else
	{
		for (int i = 0; i < nchannels; i++)
		{
			matOut[i] = Eigen::Map<EigenMatrix>(src_pointer, nrows, ncols);
			src_pointer += (nrows * ncols);
		}
	}

}

/*
By default, Eigen currently supports standard floating-point types 
(float, double, std::complex<float>, std::complex<double>, long double), 
as well as all native integer types (e.g., int, unsigned int, short, etc.), and bool.

Demo of the functions.

========== Code =============

MatlabEngWrapper mew;
mew.init();
# define EGM Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic>
mew.exec("clear all; X = [1,2;3,4]; X = uint32(X);");
mxArray* X = mew.receive("X");
EGM X_cv; matlab2eigen<unsigned int>(X, X_cv,true); mxDestroyArray(X);
EGM Y_cv = X_cv + X_cv;
mxArray* Y; eigen2matlab<unsigned int>(Y_cv, Y);
mew.send("Y", Y);

========== Output in Matlab =============

» whos,X,Y
Name      Size            Bytes  Class     Attributes

X         2x2                16  uint32
Y         2x2                16  uint32


X =

1           2
3           4


Y =

2           4
6           8




========== Code =============

MatlabEngWrapper mew;
mew.init();
mew.exec("clear all; X(:,:,1) = [1,2;3,4]; X(:,:,2) = [5,6;7,8]; X = single(X);");
mxArray* X = mew.receive("X");
# define EGM Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
vector<EGM> X_cv; matlab2eigen<float>(X, X_cv, true); mxDestroyArray(X);
vector<EGM> Y_cv = X_cv; Y_cv[0] = X_cv[0] + X_cv[0]; Y_cv[1] = X_cv[1] + X_cv[1];
mxArray* Y; eigen2matlab<float>(Y_cv, Y);
mew.send("Y", Y);

========== Output in Matlab =============

» whos X,Y
Name      Size             Bytes  Class     Attributes

X         2x2x2               32  single


Y(:,:,1) =

2     4
6     8


Y(:,:,2) =

10    12
14    16


*/


#undef EigenMatrix
#undef EIGEN_NO_DEBUG

#endif
