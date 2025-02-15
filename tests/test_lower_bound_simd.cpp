#include <iostream>
#include <vector>
#include <cassert>
#include "lower_bound_simd.hpp"

void test_lower_bound_simd() {
    // Test SIMD lower_bound with integers
    std::vector<int> vec = {1, 2, 4, 5, 6};
    std::vector<int> test_values = {3, 0, 7};
    std::vector<size_t> expected_indices = {2, 0, 5};
    for (size_t i = 0; i < test_values.size(); ++i) {
        auto it_simd = jrmwng::algorithm::simd::lower_bound(vec, test_values[i]);
        assert(it_simd == vec.begin() + expected_indices[i]);
    }

    // Test SIMD lower_bound with doubles
    std::vector<double> vec_d = {1.1, 2.2, 4.4, 5.5, 6.6};
    std::vector<double> test_values_d = {3.3, 0.0, 7.7};
    std::vector<size_t> expected_indices_d = {2, 0, 5};
    for (size_t i = 0; i < test_values_d.size(); ++i) {
        auto it_d_simd = jrmwng::algorithm::simd::lower_bound(vec_d, test_values_d[i]);
        assert(it_d_simd == vec_d.begin() + expected_indices_d[i]);
    }

    // Test SIMD lower_bound with custom predicate
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
        auto it_custom_simd = jrmwng::algorithm::simd::lower_bound(custom_vec, test_values_custom[i], std::less<CustomType>());
        assert(it_custom_simd == custom_vec.begin() + expected_indices_custom[i]);
    }

    // Test SIMD lower_bound with empty vector
    std::vector<int> empty_vec;
    auto it_empty_simd = jrmwng::algorithm::simd::lower_bound(empty_vec, 1);
    assert(it_empty_simd == empty_vec.begin()); // 1 should be inserted at index 0

    // Test SIMD lower_bound with single element vector
    std::vector<int> single_vec = {2};
    std::vector<int> test_values_single = {1, 3};
    std::vector<size_t> expected_indices_single = {0, 1};
    for (size_t i = 0; i < test_values_single.size(); ++i) {
        auto it_single_simd = jrmwng::algorithm::simd::lower_bound(single_vec, test_values_single[i]);
        assert(it_single_simd == single_vec.begin() + expected_indices_single[i]);
    }

    // Test SIMD lower_bound with all elements equal
    std::vector<int> equal_vec = {2, 2, 2, 2};
    std::vector<int> test_values_equal = {2, 1, 3};
    std::vector<size_t> expected_indices_equal = {0, 0, 4};
    for (size_t i = 0; i < test_values_equal.size(); ++i) {
        auto it_equal_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, test_values_equal[i]);
        assert(it_equal_simd == equal_vec.begin() + expected_indices_equal[i]);
    }

    // Test SIMD lower_bound with floats
    std::vector<float> vec_f = {1.1f, 2.2f, 4.4f, 5.5f, 6.6f};
    std::vector<float> test_values_f = {3.3f, 0.0f, 7.7f};
    std::vector<size_t> expected_indices_f = {2, 0, 5};
    for (size_t i = 0; i < test_values_f.size(); ++i) {
        auto it_f_simd = jrmwng::algorithm::simd::lower_bound(vec_f, test_values_f[i]);
        assert(it_f_simd == vec_f.begin() + expected_indices_f[i]);
    }

    // Test SIMD lower_bound with custom predicate for floats
    struct CustomFloatType {
        float value;
        bool operator<(const CustomFloatType& other) const {
            return value < other.value;
        }
    };

    std::vector<CustomFloatType> custom_vec_f = {{1.1f}, {3.3f}, {5.5f}};
    std::vector<CustomFloatType> test_values_custom_f = {{4.4f}, {0.0f}, {6.6f}};
    std::vector<size_t> expected_indices_custom_f = {2, 0, 3};
    for (size_t i = 0; i < test_values_custom_f.size(); ++i) {
        auto it_custom_f_simd = jrmwng::algorithm::simd::lower_bound(custom_vec_f, test_values_custom_f[i], std::less<CustomFloatType>());
        assert(it_custom_f_simd == custom_vec_f.begin() + expected_indices_custom_f[i]);
    }

    // Test SIMD lower_bound with empty vector of floats
    std::vector<float> empty_vec_f;
    auto it_empty_f_simd = jrmwng::algorithm::simd::lower_bound(empty_vec_f, 1.1f);
    assert(it_empty_f_simd == empty_vec_f.begin()); // 1.1 should be inserted at index 0

    // Test SIMD lower_bound with single element vector of floats
    std::vector<float> single_vec_f = {2.2f};
    std::vector<float> test_values_single_f = {1.1f, 3.3f};
    std::vector<size_t> expected_indices_single_f = {0, 1};
    for (size_t i = 0; i < test_values_single_f.size(); ++i) {
        auto it_single_f_simd = jrmwng::algorithm::simd::lower_bound(single_vec_f, test_values_single_f[i]);
        assert(it_single_f_simd == single_vec_f.begin() + expected_indices_single_f[i]);
    }

    // Test SIMD lower_bound with all elements equal for floats
    std::vector<float> equal_vec_f = {2.2f, 2.2f, 2.2f, 2.2f};
    std::vector<float> test_values_equal_f = {2.2f, 1.1f, 3.3f};
    std::vector<size_t> expected_indices_equal_f = {0, 0, 4};
    for (size_t i = 0; i < test_values_equal_f.size(); ++i) {
        auto it_equal_f_simd = jrmwng::algorithm::simd::lower_bound(equal_vec_f, test_values_equal_f[i]);
        assert(it_equal_f_simd == equal_vec_f.begin() + expected_indices_equal_f[i]);
    }

    // Test SIMD lower_bound with custom projection for integers
    struct CustomIntType {
        int value;
    };

    std::vector<CustomIntType> custom_vec_int = {{1}, {3}, {5}};
    std::vector<int> test_values_custom_int = {4, 0, 6};
    std::vector<size_t> expected_indices_custom_int = {2, 0, 3};
    for (size_t i = 0; i < test_values_custom_int.size(); ++i) {
        auto it_custom_proj_int = jrmwng::algorithm::simd::lower_bound(custom_vec_int, test_values_custom_int[i], std::less<int>(), [](const CustomIntType& ct) { return ct.value; });
        assert(it_custom_proj_int == custom_vec_int.begin() + expected_indices_custom_int[i]);
    }

    // Test SIMD lower_bound with custom projection for floats
    std::vector<CustomFloatType> custom_vec_float = {{1.1f}, {3.3f}, {5.5f}};
    std::vector<float> test_values_custom_float = {4.4f, 0.0f, 6.6f};
    std::vector<size_t> expected_indices_custom_float = {2, 0, 3};
    for (size_t i = 0; i < test_values_custom_float.size(); ++i) {
        auto it_custom_proj_float = jrmwng::algorithm::simd::lower_bound(custom_vec_float, test_values_custom_float[i], std::less<float>(), [](const CustomFloatType& ct) { return ct.value; });
        assert(it_custom_proj_float == custom_vec_float.begin() + expected_indices_custom_float[i]);
    }

    // Test SIMD lower_bound with custom projection for doubles
    struct CustomDoubleType {
        double value;
    };

    std::vector<CustomDoubleType> custom_vec_double = {{1.1}, {3.3}, {5.5}};
    std::vector<double> test_values_custom_double = {4.4, 0.0, 6.6};
    std::vector<size_t> expected_indices_custom_double = {2, 0, 3};
    for (size_t i = 0; i < test_values_custom_double.size(); ++i) {
        auto it_custom_proj_double = jrmwng::algorithm::simd::lower_bound(custom_vec_double, test_values_custom_double[i], std::less<double>(), [](const CustomDoubleType& ct) { return ct.value; });
        assert(it_custom_proj_double == custom_vec_double.begin() + expected_indices_custom_double[i]);
    }

    std::cout << "All SIMD tests passed!" << std::endl;
}

int main() {
    std::cout << "Compilation date and time: " << __DATE__ << " " << __TIME__ << std::endl;
    test_lower_bound_simd();
    return 0;
}
