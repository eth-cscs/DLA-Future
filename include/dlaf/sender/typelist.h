//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <type_traits>

namespace dlaf {
namespace internal {

template <typename T, typename... Ts>
struct Contains;

template <typename T>
struct Contains<T> : std::false_type {};

template <typename T, typename... Ts>
struct Contains<T, T, Ts...> : std::true_type {};

template <typename T, typename U, typename... Ts>
struct Contains<T, U, Ts...> : Contains<T, Ts...> {};

template <typename T, typename... Ts>
inline constexpr bool Contains_v = Contains<T, Ts...>::value;

template <typename PackUnique, typename PackRest>
struct UniquePackHelper;

template <template <typename...> class Pack, typename... Ts>
struct UniquePackHelper<Pack<Ts...>, Pack<>> {
  using type = Pack<Ts...>;
};

template <template <typename...> class Pack, typename... Ts, typename U, typename... Us>
struct UniquePackHelper<Pack<Ts...>, Pack<U, Us...>>
    : std::conditional<Contains_v<U, Ts...>, UniquePackHelper<Pack<Ts...>, Pack<Us...>>,
                       UniquePackHelper<Pack<Ts..., U>, Pack<Us...>>>::type {};

template <typename Pack>
struct UniquePack;

template <template <typename...> class Pack, typename... Ts>
struct UniquePack<Pack<Ts...>> : UniquePackHelper<Pack<>, Pack<Ts...>> {};

template <typename Pack>
using UniquePack_t = typename UniquePack<Pack>::type;

template <typename Pack, template <typename> class Transformer>
struct TransformPack;

template <template <typename> class Transformer, template <typename...> class Pack, typename... Ts>
struct TransformPack<Pack<Ts...>, Transformer> {
  using type = Pack<typename Transformer<Ts>::type...>;
};

template <typename Pack, template <typename> class Transformer>
using TransformPack_t = typename TransformPack<Pack, Transformer>::type;

template <typename Pack, typename T>
struct PrependPack;

template <typename T, template <typename...> class Pack, typename... Ts>
struct PrependPack<Pack<Ts...>, T> {
  using type = Pack<T, Ts...>;
};

template <typename Pack, typename T>
using PrependPack_t = typename PrependPack<Pack, T>::type;
}
}
