#pragma once

#include <functional>     // For std::less, std::identity, std::invoke
#include <ranges>         // For std::ranges::forward_range, std::ranges::iterator_t, std::ranges::begin, std::ranges::end
#include <iterator>       // For std::distance
#include <type_traits>    // For std::is_same_v, std::make_index_sequence
#include <bit>            // For std::popcount
#include <utility>        // For std::index_sequence

/**
 * @file lower_bound.hpp
 * @brief Provides implementations of the lower_bound algorithm for different use cases.
 * 
 * This file contains template functions to find the first position in a sorted range where a given value could be inserted without violating the order.
 * It includes overloads for iterators and ranges, with optional custom comparison and projection functions.
 */

namespace jrmwng
{
    namespace algorithm
    {
        /**
         * @brief Finds the first position in a sorted range where a given value could be inserted without violating the order.
         * 
         * @tparam Titerator The type of the iterator.
         * @tparam T The type of the value to compare.
         * @tparam Tpredicate The type of the predicate function.
         * @param first The beginning of the range.
         * @param last The end of the range.
         * @param t The value to compare.
         * @param pred The predicate function that returns true if the first argument is less than the second.
         * @return Titerator The iterator pointing to the first position where the value could be inserted.
         * 
         * @example
         * std::vector<int> vec = {1, 2, 4, 5, 6};
         * auto it = jrmwng::algorithm::lower_bound(vec.begin(), vec.end(), 3, std::less<int>());
         * // it points to vec.begin() + 2
         */
        template <typename Titerator, typename T, typename Tpredicate>
        Titerator lower_bound(Titerator first, Titerator last, T const & t, Tpredicate pred)
        {
            while (first < last)
            {
                Titerator mid = first + (last - first) / 2;
                if (pred(*mid, t))
                {
                    first = mid + 1;
                }
                else
                {
                    last = mid;
                }
            }
            return first;
        }

        /**
         * @brief Finds the first position in a sorted range where a given value could be inserted without violating the order.
         * 
         * @tparam Titerator The type of the iterator.
         * @tparam T The type of the value to compare.
         * @param first The beginning of the range.
         * @param last The end of the range.
         * @param t The value to compare.
         * @return Titerator The iterator pointing to the first position where the value could be inserted.
         * 
         * @example
         * std::vector<int> vec = {1, 2, 4, 5, 6};
         * auto it = jrmwng::algorithm::lower_bound(vec.begin(), vec.end(), 3);
         * // it points to vec.begin() + 2
         */
        template <typename Titerator, typename T>
        Titerator lower_bound(Titerator first, Titerator last, T const & t)
        {
            return jrmwng::algorithm::lower_bound(first, last, t, std::less<T>());
        }

        namespace ranges
        {
            /**
             * @brief Performs a lower bound search on a range using n-ary search.
             * 
             * @tparam Range The type of the range.
             * @tparam T The type of the value to search for.
             * @tparam Compare The type of the comparison function.
             * @tparam Projection The type of the projection function.
             * @tparam zuPARTITION_i The partition indices.
             * @param r The range to search.
             * @param value The value to search for.
             * @param comp The comparison function.
             * @param proj The projection function.
             * @return An iterator to the lower bound of the value in the range.
             * 
             * @details The comparison function should return a bitmask indicating partitions that satisfy the comparison.
             */
            template <typename Range, typename T, typename Compare, typename Projection, size_t... zuPARTITION_i>
            requires std::ranges::forward_range<Range>
            std::ranges::iterator_t<Range> lower_bound(Range & r, const T& value, Compare comp, Projection proj, std::index_sequence<zuPARTITION_i...>)
            {
                static_assert(std::is_same_v<std::index_sequence<zuPARTITION_i...>, std::make_index_sequence<sizeof...(zuPARTITION_i)>>, "Invalid index sequence");
//                static_assert(sizeof...(zuPARTITION_i) == 1 || std::is_same_v<int, decltype(comp(value, value))>, "Invalid comparison function");

                using Titerator = std::ranges::iterator_t<Range>;

                Titerator first = std::ranges::begin(r); // Initialize the first iterator
                Titerator last = std::ranges::end(r); // Initialize the last iterator
        
                while (first != last) // Loop until the range is exhausted
                {
                    // Create an array to hold iterators at partition points
                    Titerator const iters[]
                    {
                        ((first + (1 + zuPARTITION_i) * std::distance(first, last) / (1 + sizeof...(zuPARTITION_i))))...
                    };
        
                    auto const nCompare = std::invoke(comp, std::invoke(proj, (*iters[zuPARTITION_i])...), value);
                    static_assert(sizeof(nCompare) <= 4, "Invalid comparison function");

                    // 1-based index of the last partition point that satisfies the comparison.
                    // zero if neither partition point satisfies the comparison.
                    int const nIndex1 = std::popcount((unsigned)nCompare);
                    // 0-based index of the last partition point that satisfies the comparison
                    int const nIndex0 = nIndex1 - 1;
        
                    if (nIndex1) // If any partition point satisfies the comparison
                    {
                        // Move the first iterator to the right of the last partition point that satisfies the comparison.
                        first = iters[nIndex0] + 1;
                    }
                    else
                    {
                        // NOP: first, *value*, parition_point_1, partition_point_2, ..., partition_point_n, last
                    }
                    if (nIndex1 < sizeof...(zuPARTITION_i))
                    {
                        // Move the last iterator to the left of the partition point that is next to the last partition point that satisfies the comparison.
                        last = iters[nIndex1];
                    }
                    else
                    {
                        // NOP: first, partition_point_1, partition_point_2, ..., partition_point_n, *value*, last
                    }
                }
        
                return first; // Return the iterator to the lower bound
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
             * auto it = jrmwng::algorithm::lower_bound(vec, 3, std::less<int>(), [](int x) { return x; });
             * // it points to vec.begin() + 2
             */
            template <typename Range, typename T, typename Compare = std::less<T>, typename Projection = std::identity>
            requires std::ranges::forward_range<Range>
            std::ranges::iterator_t<Range> lower_bound(Range & r, T const & value, Compare comp = {}, Projection proj = {})
            {
                return jrmwng::algorithm::ranges::lower_bound(r, value, comp, proj, std::make_index_sequence<1>{});
            }
        }
    }
}
