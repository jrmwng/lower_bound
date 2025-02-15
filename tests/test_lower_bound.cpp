#include <iostream>
#include <vector>
#include <cassert>
#include "lower_bound.hpp"

void test_lower_bound() {
    // Test with integers
    std::vector<int> vec = {1, 2, 4, 5, 6};
    std::vector<int> test_values = {3, 0, 7};
    std::vector<size_t> expected_indices = {2, 0, 5};
    for (size_t i = 0; i < test_values.size(); ++i) {
        auto it = jrmwng::algorithm::lower_bound(vec.begin(), vec.end(), test_values[i], std::less<int>());
        assert(it == vec.begin() + expected_indices[i]);
    }

    // Test with doubles
    std::vector<double> vec_d = {1.1, 2.2, 4.4, 5.5, 6.6};
    std::vector<double> test_values_d = {3.3, 0.0, 7.7};
    std::vector<size_t> expected_indices_d = {2, 0, 5};
    for (size_t i = 0; i < test_values_d.size(); ++i) {
        auto it_d = jrmwng::algorithm::lower_bound(vec_d.begin(), vec_d.end(), test_values_d[i], std::less<double>());
        assert(it_d == vec_d.begin() + expected_indices_d[i]);
    }

    // Test with custom predicate
    struct CustomType {
        int value;
        bool operator<(const CustomType& other) const {
            return value < other.value;
        }
    };

    std::vector<CustomType> custom_vec = {{1}, {3}, {5}};
    std::vector<CustomType> test_values_custom = {{4}, {0}, {6}};
    std::vector<size_t> expected_indices_custom = {2, 0, 3};
    for (size_t i = 0; i < test_values_custom.size(); ++i) {
        auto it_custom = jrmwng::algorithm::lower_bound(custom_vec.begin(), custom_vec.end(), test_values_custom[i], std::less<CustomType>());
        assert(it_custom == custom_vec.begin() + expected_indices_custom[i]);
    }

    // Test with empty vector
    std::vector<int> empty_vec;
    auto it_empty = jrmwng::algorithm::lower_bound(empty_vec.begin(), empty_vec.end(), 1);
    assert(it_empty == empty_vec.begin()); // 1 should be inserted at index 0

    // Test with single element vector
    std::vector<int> single_vec = {2};
    std::vector<int> test_values_single = {1, 3};
    std::vector<size_t> expected_indices_single = {0, 1};
    for (size_t i = 0; i < test_values_single.size(); ++i) {
        auto it_single = jrmwng::algorithm::lower_bound(single_vec.begin(), single_vec.end(), test_values_single[i]);
        assert(it_single == single_vec.begin() + expected_indices_single[i]);
    }

    // Test with all elements equal
    std::vector<int> equal_vec = {2, 2, 2, 2};
    std::vector<int> test_values_equal = {2, 1, 3};
    std::vector<size_t> expected_indices_equal = {0, 0, 4};
    for (size_t i = 0; i < test_values_equal.size(); ++i) {
        auto it_equal = jrmwng::algorithm::lower_bound(equal_vec.begin(), equal_vec.end(), test_values_equal[i]);
        assert(it_equal == equal_vec.begin() + expected_indices_equal[i]);
    }

    // Test range-based lower_bound with integers
    for (size_t i = 0; i < test_values.size(); ++i) {
        auto it_range = jrmwng::algorithm::ranges::lower_bound(vec, test_values[i], std::less<int>(), [](int i) { return i; });
        assert(it_range == vec.begin() + expected_indices[i]);
    }

    // Test range-based lower_bound with doubles
    for (size_t i = 0; i < test_values_d.size(); ++i) {
        auto it_d_range = jrmwng::algorithm::ranges::lower_bound(vec_d, test_values_d[i], std::less<double>(), [](double d) { return d; });
        assert(it_d_range == vec_d.begin() + expected_indices_d[i]);
    }

    // Test range-based lower_bound with custom predicate
    for (size_t i = 0; i < test_values_custom.size(); ++i) {
        auto it_custom_range = jrmwng::algorithm::ranges::lower_bound(custom_vec, test_values_custom[i], std::less<CustomType>(), [](const CustomType& ct) { return ct; });
        assert(it_custom_range == custom_vec.begin() + expected_indices_custom[i]);
    }

    // Test range-based lower_bound with empty vector
    auto it_empty_range = jrmwng::algorithm::ranges::lower_bound(empty_vec, 1, std::less<int>(), [](int i) { return i; });
    assert(it_empty_range == empty_vec.begin()); // 1 should be inserted at index 0

    // Test range-based lower_bound with single element vector
    for (size_t i = 0; i < test_values_single.size(); ++i) {
        auto it_single_range = jrmwng::algorithm::ranges::lower_bound(single_vec, test_values_single[i], std::less<int>(), [](int i) { return i; });
        assert(it_single_range == single_vec.begin() + expected_indices_single[i]);
    }

    // Test range-based lower_bound with all elements equal
    for (size_t i = 0; i < test_values_equal.size(); ++i) {
        auto it_equal_range = jrmwng::algorithm::ranges::lower_bound(equal_vec, test_values_equal[i], std::less<int>(), [](int i) { return i; });
        assert(it_equal_range == equal_vec.begin() + expected_indices_equal[i]);
    }

    // Test default arguments with integers
    for (size_t i = 0; i < test_values.size(); ++i) {
        auto it_default = jrmwng::algorithm::lower_bound(vec.begin(), vec.end(), test_values[i]);
        assert(it_default == vec.begin() + expected_indices[i]);
    }

    // Test default arguments with doubles
    for (size_t i = 0; i < test_values_d.size(); ++i) {
        auto it_d_default = jrmwng::algorithm::lower_bound(vec_d.begin(), vec_d.end(), test_values_d[i]);
        assert(it_d_default == vec_d.begin() + expected_indices_d[i]);
    }

    // Test default arguments with custom predicate
    for (size_t i = 0; i < test_values_custom.size(); ++i) {
        auto it_custom_default = jrmwng::algorithm::lower_bound(custom_vec.begin(), custom_vec.end(), test_values_custom[i]);
        assert(it_custom_default == custom_vec.begin() + expected_indices_custom[i]);
    }

    // Test default arguments with empty vector
    auto it_empty_default = jrmwng::algorithm::lower_bound(empty_vec.begin(), empty_vec.end(), 1);
    assert(it_empty_default == empty_vec.begin()); // 1 should be inserted at index 0

    // Test default arguments with single element vector
    for (size_t i = 0; i < test_values_single.size(); ++i) {
        auto it_single_default = jrmwng::algorithm::lower_bound(single_vec.begin(), single_vec.end(), test_values_single[i]);
        assert(it_single_default == single_vec.begin() + expected_indices_single[i]);
    }

    // Test default arguments with all elements equal
    for (size_t i = 0; i < test_values_equal.size(); ++i) {
        auto it_equal_default = jrmwng::algorithm::lower_bound(equal_vec.begin(), equal_vec.end(), test_values_equal[i]);
        assert(it_equal_default == equal_vec.begin() + expected_indices_equal[i]);
    }

    // Test range-based lower_bound with default arguments
    for (size_t i = 0; i < test_values.size(); ++i) {
        auto it_range_default = jrmwng::algorithm::ranges::lower_bound(vec, test_values[i]);
        assert(it_range_default == vec.begin() + expected_indices[i]);
    }

    for (size_t i = 0; i < test_values_d.size(); ++i) {
        auto it_d_range_default = jrmwng::algorithm::ranges::lower_bound(vec_d, test_values_d[i]);
        assert(it_d_range_default == vec_d.begin() + expected_indices_d[i]);
    }

    for (size_t i = 0; i < test_values_custom.size(); ++i) {
        auto it_custom_range_default = jrmwng::algorithm::ranges::lower_bound(custom_vec, test_values_custom[i]);
        assert(it_custom_range_default == custom_vec.begin() + expected_indices_custom[i]);
    }

    auto it_empty_range_default = jrmwng::algorithm::ranges::lower_bound(empty_vec, 1);
    assert(it_empty_range_default == empty_vec.begin()); // 1 should be inserted at index 0

    for (size_t i = 0; i < test_values_single.size(); ++i) {
        auto it_single_range_default = jrmwng::algorithm::ranges::lower_bound(single_vec, test_values_single[i]);
        assert(it_single_range_default == single_vec.begin() + expected_indices_single[i]);
    }

    for (size_t i = 0; i < test_values_equal.size(); ++i) {
        auto it_equal_range_default = jrmwng::algorithm::ranges::lower_bound(equal_vec, test_values_equal[i]);
        assert(it_equal_range_default == equal_vec.begin() + expected_indices_equal[i]);
    }

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    std::cout << "Compilation date and time: " << __DATE__ << " " << __TIME__ << std::endl;
    test_lower_bound();
    return 0;
}