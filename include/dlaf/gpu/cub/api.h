//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2020, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#if defined(DLAF_WITH_HIP)

#include "rocprim/rocprim.hpp"
#include <whip.hpp>

namespace cub {

    template<
        typename ValueType,
        typename ConversionOp,
        typename InputIteratorT,
        typename OffsetT = std::ptrdiff_t // ignored
    >
    using TransformInputIterator = ::rocprim::transform_iterator<InputIteratorT, ConversionOp, ValueType>;

    template<
        typename ValueType,
        typename OffsetT = std::ptrdiff_t
    >
    using CountingInputIterator = ::rocprim::counting_iterator<ValueType, OffsetT>;

    namespace detail
    {

    	// CUB uses value_type of OutputIteratorT (if not void) as a type of intermediate results in reduce,
    	// for example:
    	//
    	// rocPRIM (as well as Thrust) uses result type of BinaryFunction instead (if not void):
	//
	// using input_type = typename std::iterator_traits<InputIterator>::value_type;
	// using result_type = typename ::rocprim::detail::match_result_type<
	//     input_type, BinaryFunction
	// >::type;
	//
	// For short -> float using Sum()
	// CUB:     float Sum(float, float)
	// rocPRIM: short Sum(short, short)
	//
	// This wrapper allows to have compatibility with CUB in hipCUB.
	template<
	    class InputIteratorT,
	    class OutputIteratorT,
	    class BinaryFunction
	>
	struct convert_result_type_wrapper
	{
	    using input_type = typename std::iterator_traits<InputIteratorT>::value_type;
	    using output_type = typename std::iterator_traits<OutputIteratorT>::value_type;
	    using result_type =
	        typename std::conditional<
	            std::is_void<output_type>::value, input_type, output_type
	        >::type;

	    convert_result_type_wrapper(BinaryFunction op) : op(op) {}

	    template<class T>
	    __host__ __device__ inline
	    constexpr result_type operator()(const T &a, const T &b) const
	    {
	        return static_cast<result_type>(op(a, b));
	    }

	    BinaryFunction op;
	};

	template<
	    class InputIteratorT,
	    class OutputIteratorT,
	    class BinaryFunction
	>
	inline
	convert_result_type_wrapper<InputIteratorT, OutputIteratorT, BinaryFunction>
	convert_result_type(BinaryFunction op)
	{
	    return convert_result_type_wrapper<InputIteratorT, OutputIteratorT, BinaryFunction>(op);
	}

    } // end detail namespace

    struct Sum
    {
        template<class T>
        __host__ __device__ inline
        constexpr T operator()(const T &a, const T &b) const
        {
            return a + b;
        }
    };

    class DeviceReduce
    {
    public:
    	template <
    	    typename InputIteratorT,
    	    typename OutputIteratorT,
    	    typename ReduceOpT,
    	    typename T
    	>
    	__host__ static
    	whip::error_t Reduce(void *d_temp_storage,
    	                  std::size_t &temp_storage_bytes,
    	                  InputIteratorT d_in,
    	                  OutputIteratorT d_out,
    	                  int num_items,
    	                  ReduceOpT reduction_op,
    	                  T init,
    	                  whip::stream_t stream = 0,
    	                  bool debug_synchronous = false)
    	{
    	    return ::rocprim::reduce(
    	        d_temp_storage, temp_storage_bytes,
    	        d_in, d_out, init, num_items,
    	        ::cub::detail::convert_result_type<InputIteratorT, OutputIteratorT>(reduction_op),
    	        stream, debug_synchronous
    	    );
    	}
        template <
            typename InputIteratorT,
            typename OutputIteratorT
        >
        __host__ static
        whip::error_t Sum(void *d_temp_storage,
                       std::size_t &temp_storage_bytes,
                       InputIteratorT d_in,
                       OutputIteratorT d_out,
                       int num_items,
                       whip::stream_t stream = 0,
                       bool debug_synchronous = false)
        {
            using T = typename std::iterator_traits<InputIteratorT>::value_type;
            return Reduce(
                d_temp_storage, temp_storage_bytes,
                d_in, d_out, num_items, ::cub::Sum(), T(0),
                stream, debug_synchronous
            );
        }

    };

    struct DeviceSegmentedReduce
    {
        template<
            typename InputIteratorT,
            typename OutputIteratorT,
            typename OffsetIteratorT,
            typename ReductionOp,
            typename T
        >
        __host__ static
        hipError_t Reduce(void * d_temp_storage,
                          std::size_t& temp_storage_bytes,
                          InputIteratorT d_in,
                          OutputIteratorT d_out,
                          int num_segments,
                          OffsetIteratorT d_begin_offsets,
                          OffsetIteratorT d_end_offsets,
                          ReductionOp reduction_op,
                          T initial_value,
                          whip::stream_t stream = 0,
                          bool debug_synchronous = false)
        {
            return ::rocprim::segmented_reduce(
                d_temp_storage, temp_storage_bytes,
                d_in, d_out,
                num_segments, d_begin_offsets, d_end_offsets,
                ::cub::detail::convert_result_type<InputIteratorT, OutputIteratorT>(reduction_op),
                initial_value,
                stream, debug_synchronous
            );
        }
    };

}

#else

#include <cub/cub.cuh>

#endif
