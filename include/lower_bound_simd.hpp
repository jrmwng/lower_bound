#pragma once

#include "lower_bound.hpp"  // Project-specific header for lower_bound functionality

#include <immintrin.h>      // for __m256, __m256i, __m256d and associated intrinsics
#include <utility>          // for std::make_index_sequence, std::index_sequence
#include <functional>       // for std::invoke, std::less, std::less_equal, std::greater, std::greater_equal, std::identity
#include <type_traits>      // for std::is_invocable_v, std::is_invocable_r_v
#include <ranges>           // for std::ranges::forward_range, std::ranges::iterator_t

/**
 * @file lower_bound_simd.hpp
 * @brief Provides SIMD-optimized implementations of the lower_bound algorithm for different data types.
 * 
 * This file contains template functions and specializations to find the first position in a sorted range where a given value could be inserted without violating the order.
 * It includes SIMD-optimized versions for float, double, and int types.
 */

namespace jrmwng
{
    namespace algorithm
    {
        namespace simd
        {
            namespace details
            {
                /**
                 * @brief Traits for SIMD operations.
                 * 
                 * @tparam T The type of the value.
                 */
                template <typename T>
                struct simd_traits;

                /**
                 * @brief Specialization of simd_traits for float type.
                 */
                template <>
                struct simd_traits<float>
                {
                    using simd_type = __m256;
                    using index_sequence_type = std::make_index_sequence<8>;
                    constexpr static size_t simd_size_v = 8;

                    static __m256 set1(float const rValue)
                    {
                        return _mm256_set1_ps(rValue);
                    }
                    static __m256 setr(float const r0, float const r1, float const r2, float const r3, float const r4, float const r5, float const r6, float const r7)
                    {
                        return _mm256_setr_ps(r0, r1, r2, r3, r4, r5, r6, r7);
                    }
                    static int cmp_lt(__m256 const &lhs, __m256 const &rhs)
                    {
                        return _mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ));
                    }
                    static int cmp_le(__m256 const &lhs, __m256 const &rhs)
                    {
                        return _mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ));
                    }
                    static int cmp_gt(__m256 const &lhs, __m256 const &rhs)
                    {
                        return _mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ));
                    }
                    static int cmp_ge(__m256 const &lhs, __m256 const &rhs)
                    {
                        return _mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_GE_OQ));
                    }
                    template <int nINDEX>
                    static float extract(__m256 const &lhs)
                    {
                        return _mm256_extract_ps(lhs, nINDEX);
                    }
                };

                /**
                 * @brief Specialization of simd_traits for double type.
                 */
                template <>
                struct simd_traits<double>
                {
                    using simd_type = __m256d;
                    using index_sequence_type = std::make_index_sequence<4>;
                    constexpr static size_t simd_size_v = 4;

                    static __m256d set1(double const dValue)
                    {
                        return _mm256_set1_pd(dValue);
                    }
                    static __m256d setr(double const d0, double const d1, double const d2, double const d3)
                    {
                        return _mm256_setr_pd(d0, d1, d2, d3);
                    }
                    static int cmp_lt(__m256d const &lhs, __m256d const &rhs)
                    {
                        return _mm256_movemask_pd(_mm256_cmp_pd(lhs, rhs, _CMP_LT_OQ));
                    }
                    static int cmp_le(__m256d const &lhs, __m256d const &rhs)
                    {
                        return _mm256_movemask_pd(_mm256_cmp_pd(lhs, rhs, _CMP_LE_OQ));
                    }
                    static int cmp_gt(__m256d const &lhs, __m256d const &rhs)
                    {
                        return _mm256_movemask_pd(_mm256_cmp_pd(lhs, rhs, _CMP_GT_OQ));
                    }
                    static int cmp_ge(__m256d const &lhs, __m256d const &rhs)
                    {
                        return _mm256_movemask_pd(_mm256_cmp_pd(lhs, rhs, _CMP_GE_OQ));
                    }
                    template <int nINDEX>
                    static double extract(__m256d const &lhs)
                    {
                        return _mm256_extract_pd(lhs, nINDEX);
                    }
                };

                /**
                 * @brief Specialization of simd_traits for int type.
                 */
                template <>
                struct simd_traits<int>
                {
                    using simd_type = __m256i;
                    using index_sequence_type = std::make_index_sequence<8>;
                    constexpr static size_t simd_size_v = 8;

                    static __m256i set1(int const nValue)
                    {
                        return _mm256_set1_epi32(nValue);
                    }
                    static __m256i setr(int const n0, int const n1, int const n2, int const n3, int const n4, int const n5, int const n6, int const n7)
                    {
                        return _mm256_setr_epi32(n0, n1, n2, n3, n4, n5, n6, n7);
                    }
                    static int cmp_lt(__m256i const &lhs, __m256i const &rhs)
                    {
                        return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(rhs, lhs)));
                    }
                    static int cmp_le(__m256i const &lhs, __m256i const &rhs)
                    {
                        return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(rhs, lhs)));
                    }
                    static int cmp_gt(__m256i const &lhs, __m256i const &rhs)
                    {
                        return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(lhs, rhs)));
                    }
                    static int cmp_ge(__m256i const &lhs, __m256i const &rhs)
                    {
                        return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(lhs, rhs)));
                    }
                    template <int nINDEX>
                    static int extract(__m256i const &lhs)
                    {
                        return _mm256_extract_epi32(lhs, nINDEX);
                    }
                };
                
                /**
                 * @brief Comparison function for SIMD types.
                 * 
                 * @tparam Tcompare The type of the comparison function.
                 * @tparam T The type of the value.
                 */
                template <typename Tcompare, typename T>
                struct simd_compare_t
                {
                    static_assert(std::is_invocable_v<Tcompare, T, T>, "Invalid comparison function");

                    using simd_type = typename simd_traits<T>::simd_type;

                    Tcompare compare;

                    /**
                     * @brief Compares two scalar values using the comparison function.
                     * 
                     * @param lhs The left-hand side value.
                     * @param rhs The right-hand side value.
                     * @return int The result of the comparison.
                     */
                    int operator() (T const lhs, T const rhs) const
                    {
                        return std::invoke(compare, lhs, rhs);
                    }

                    /**
                     * @brief Applies the comparison function to each element of the SIMD vector and a scalar value.
                     * 
                     * @tparam zuELEMENT_i The indices of the elements in the SIMD vector.
                     * @param lhs The SIMD vector.
                     * @param tRHS The scalar value.
                     * @param std::index_sequence<zuELEMENT_i...> The index sequence.
                     * @return int The result of the comparison.
                     */
                    template <size_t... zuELEMENT_i>
                    int apply(simd_type const &lhs, T const &tRHS, std::index_sequence<zuELEMENT_i...>) const
                    {
                        return
                            ((std::invoke(compare, simd_traits<T>::extract<zuELEMENT_i>(lhs), tRHS) ? (0x01 << zuELEMENT_i) : 0) | ... | 0);
                    }

                    /**
                     * @brief Checks if the comparison function can be applied to SIMD types.
                     */
                    constexpr static bool is_simd_compare_v = std::is_invocable_r_v<int, Tcompare, simd_type, simd_type>
                        || std::is_same_v<Tcompare, std::less<T>>
                        || std::is_same_v<Tcompare, std::less_equal<T>>
                        || std::is_same_v<Tcompare, std::greater<T>>
                        || std::is_same_v<Tcompare, std::greater_equal<T>>;

                    /**
                     * @brief Compares a SIMD vector and a scalar value using the comparison function.
                     * 
                     * @param lhs The SIMD vector.
                     * @param tRHS The scalar value.
                     * @return int The result of the comparison.
                     */
                    int operator() (simd_type const &lhs, T const &tRHS) const
                    {
                        if constexpr (is_simd_compare_v)
                        {
                            if constexpr (std::is_invocable_r_v<int, Tcompare, simd_type, simd_type>)
                            {
                                return std::invoke(compare, lhs, simd_traits<T>::set1(tRHS));
                            }
                            else if constexpr (std::is_same_v<Tcompare, std::less<T>>)
                            {
                                return simd_traits<T>::cmp_lt(lhs, simd_traits<T>::set1(tRHS));
                            }
                            else if constexpr (std::is_same_v<Tcompare, std::less_equal<T>>)
                            {
                                return simd_traits<T>::cmp_le(lhs, simd_traits<T>::set1(tRHS));
                            }
                            else if constexpr (std::is_same_v<Tcompare, std::greater<T>>)
                            {
                                return simd_traits<T>::cmp_gt(lhs, simd_traits<T>::set1(tRHS));
                            }
                            else if constexpr (std::is_same_v<Tcompare, std::greater_equal<T>>)
                            {
                                return simd_traits<T>::cmp_ge(lhs, simd_traits<T>::set1(tRHS));
                            }
                            else
                            {
                                static_assert(false, "Inconsistent `is_simd_compare_v`");
                            }
                        }
                        else
                        {
                            return apply(lhs, tRHS, typename simd_traits<T>::index_sequence_type{});
                        }
                    }

                    /**
                     * @brief Applies the comparison function to each element of a tuple and a scalar value.
                     * 
                     * @tparam Ts The types of the elements in the tuple.
                     * @tparam zuOFFSET The offset in the tuple.
                     * @tparam zuELEMENT_i The indices of the elements in the tuple.
                     * @param tupleLHS The tuple.
                     * @param tRHS The scalar value.
                     * @return int The result of the comparison.
                     */
                    template <typename... Ts, size_t zuOFFSET, size_t... zuELEMENT_i>
                    int apply(std::tuple<Ts...> const &tupleLHS, T const &tRHS, std::integral_constant<size_t, zuOFFSET>, std::index_sequence<zuELEMENT_i...>) const
                    {
                        if constexpr (std::is_same_v<std::index_sequence<zuELEMENT_i...>, std::make_index_sequence<8>> && is_simd_compare_v)
                        {
                            return operator()(simd_traits<T>::setr(std::get<zuOFFSET + zuELEMENT_i>(tupleLHS)...), tRHS) << zuOFFSET;
                        }
                        else
                        {
                            return ((std::invoke(compare, std::get<zuOFFSET + zuELEMENT_i>(tupleLHS), tRHS) ? (0x01 << (zuOFFSET + zuELEMENT_i)) : 0) | ... | 0);
                        }
                    }

                    /**
                     * @brief Applies the comparison function to each element of a tuple and a scalar value.
                     * 
                     * @tparam Ts The types of the elements in the tuple.
                     * @tparam zuOFFSET8 The offsets in the tuple.
                     * @param tupleLHS The tuple.
                     * @param tRHS The scalar value.
                     * @return int The result of the comparison.
                     */
                    template <typename... Ts, size_t... zuOFFSET8>
                    int apply(std::tuple<Ts...> const &tupleLHS, T const &tRHS, std::index_sequence<zuOFFSET8...>) const
                    {
                        return (0 | ... | apply(tupleLHS, tRHS, std::integral_constant<size_t, zuOFFSET8 * simd_traits<T>::simd_size_v>{}, std::make_index_sequence<std::min(simd_traits<T>::simd_size_v, sizeof...(Ts) - (zuOFFSET8 * simd_traits<T>::simd_size_v))>{}));
                    }

                    /**
                     * @brief Compares each element of a tuple and a scalar value using the comparison function.
                     * 
                     * @tparam Ts The types of the elements in the tuple.
                     * @param tupleLHS The tuple.
                     * @param tRHS The scalar value.
                     * @return int The result of the comparison.
                     */
                    template <typename... Ts>
                    int operator() (std::tuple<Ts...> const &tupleLHS, T const &tRHS) const
                    {
                        static_assert(sizeof...(Ts) <= 32, "Invalid tuple size");

                        return apply(tupleLHS, tRHS, std::make_index_sequence<(sizeof...(Ts) + (simd_traits<T>::simd_size_v-size_t(1)))/simd_traits<T>::simd_size_v>{});
                    }
                };

                /**
                 * @brief Projection function for SIMD types.
                 * 
                 * @tparam Tprojection The type of the projection function.
                 */
                template <typename Tprojection>
                struct simd_projection_t
                {
                    Tprojection projection;

                    template <typename... Targs>
                    auto operator()(Targs ... args) const
                    {
                        if constexpr (sizeof...(Targs) == 1)
                        {
                            return std::invoke(projection, args...);
                        }
                        else if constexpr (sizeof...(Targs) == 8 && (std::is_same_v<Targs, float> && ... && true) && std::is_invocable_v<Tprojection, __m256>)
                        {
                            return std::invoke(projection, _mm256_setr_ps(args...));
                        }
                        else if constexpr (sizeof...(Targs) == 8 && (std::is_same_v<Targs, int> && ... && true) && std::is_invocable_v<Tprojection, __m256i>)
                        {
                            return std::invoke(projection, _mm256_setr_epi32(args...));
                        }
                        else if constexpr (sizeof...(Targs) == 4 && (std::is_same_v<Targs, double> && ... && true) && std::is_invocable_v<Tprojection, __m256d>)
                        {
                            return std::invoke(projection, _mm256_setr_pd(args...));
                        }
                        else
                        {
                            return std::make_tuple(std::invoke(projection, args)...);
                        }
                    }
                };
            }

            /**
             * @brief Finds the first position in a sorted range where a given value could be inserted without violating the order.
             * 
             * @tparam Range The type of the range.
             * @tparam T The type of the value to compare.
             * @tparam Compare The type of the comparison function.
             * @tparam Projection The type of the projection function.
             * @param r The range to search.
             * @param value The value to compare.
             * @param comp The comparison function.
             * @param proj The projection function.
             * @return auto The iterator pointing to the first position where the value could be inserted.
             * 
             * @example
             * std::vector<int> vec = {1, 2, 4, 5, 6};
             * auto it = jrmwng::algorithm::simd::lower_bound(vec, 3, std::less<int>(), [](int x) { return x; });
             * // it points to vec.begin() + 2
             */
            template <typename Range, typename T, typename Compare = std::less<T>, typename Projection = std::identity>
            requires std::ranges::forward_range<Range>
            std::ranges::iterator_t<Range> lower_bound(Range && r, T const & value, Compare comp = {}, Projection proj = {})
            {
                using Tinput = typename std::ranges::range_value_t<Range>;

                /**
                 * @brief Specialization for float and int types.
                 */
                if constexpr (std::is_same_v<float, T> || std::is_same_v<int, T>)
                {
                    if constexpr (std::is_invocable_v<Projection, Tinput, Tinput, Tinput, Tinput, Tinput, Tinput, Tinput, Tinput>)
                    {
                        return jrmwng::algorithm::ranges::lower_bound(r, value, details::simd_compare_t<Compare, T>{comp}, details::simd_projection_t<Projection>{proj}, std::make_index_sequence<8>{});
                    }
                    else if constexpr (std::is_invocable_v<Projection, __m256>)
                    {
                        return jrmwng::algorithm::ranges::lower_bound(r, value, details::simd_compare_t<Compare, T>{comp}, details::simd_projection_t<Projection>{proj}, std::make_index_sequence<8>{});
                    }
                    else if constexpr (std::is_invocable_v<Projection, __m256i>)
                    {
                        return jrmwng::algorithm::ranges::lower_bound(r, value, details::simd_compare_t<Compare, T>{comp}, details::simd_projection_t<Projection>{proj}, std::make_index_sequence<8>{});
                    }
                    else
                    {
                        return jrmwng::algorithm::ranges::lower_bound(r, value, comp, proj, std::make_index_sequence<1>{});
                    }
                }
                /**
                 * @brief Specialization for double type.
                 */
                else if constexpr (std::is_same_v<double, T>)
                {
                    if constexpr (std::is_invocable_v<Projection, Tinput, Tinput, Tinput, Tinput>)
                    {
                        return jrmwng::algorithm::ranges::lower_bound(r, value, details::simd_compare_t<Compare, T>{comp}, details::simd_projection_t<Projection>{proj}, std::make_index_sequence<4>{});
                    }
                    else if constexpr (std::is_invocable_v<Projection, __m256d>)
                    {
                        return jrmwng::algorithm::ranges::lower_bound(r, value, details::simd_compare_t<Compare, T>{comp}, details::simd_projection_t<Projection>{proj}, std::make_index_sequence<4>{});
                    }
                    else
                    {
                        return jrmwng::algorithm::ranges::lower_bound(r, value, comp, proj, std::make_index_sequence<1>{});
                    }
                }
                /**
                 * @brief Default case for other types.
                 */
                else
                {
                    return jrmwng::algorithm::ranges::lower_bound(r, value, comp, proj, std::make_index_sequence<1>{});
                }
            }

        }
    }
}