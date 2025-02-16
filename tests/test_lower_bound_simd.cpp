#include <iostream>
#include <vector>
#include <cassert>
#include "lower_bound_simd.hpp"

void test_lower_bound_simd() {
    // Test SIMD lower_bound with integers
    std::vector<int> vec = {1, 2, 4, 5, 6};
    auto it_simd = jrmwng::algorithm::simd::lower_bound(vec, 3);
    assert(it_simd == vec.begin() + 2); // 3 should be inserted at index 2

    // Test SIMD lower_bound with doubles
    std::vector<double> vec_d = {1.1, 2.2, 4.4, 5.5, 6.6};
    auto it_d_simd = jrmwng::algorithm::simd::lower_bound(vec_d, 3.3);
    assert(it_d_simd == vec_d.begin() + 2); // 3.3 should be inserted at index 2

    // Test SIMD lower_bound with custom predicate
    struct CustomType {
        int value;
        bool operator<(const CustomType& other) const {
            return value < other.value;
        }
    };

    std::vector<CustomType> custom_vec = {{1}, {3}, {5}};
    CustomType new_value = {4};
    auto it_custom_simd = jrmwng::algorithm::simd::lower_bound(custom_vec, new_value, std::less<CustomType>());
    assert(it_custom_simd == custom_vec.begin() + 2); // {4} should be inserted at index 2

    // Test SIMD lower_bound with empty vector
    std::vector<int> empty_vec;
    auto it_empty_simd = jrmwng::algorithm::simd::lower_bound(empty_vec, 1);
    assert(it_empty_simd == empty_vec.begin()); // 1 should be inserted at index 0

    // Test SIMD lower_bound with single element vector
    std::vector<int> single_vec = {2};
    auto it_single_simd = jrmwng::algorithm::simd::lower_bound(single_vec, 1);
    assert(it_single_simd == single_vec.begin()); // 1 should be inserted at index 0

    it_single_simd = jrmwng::algorithm::simd::lower_bound(single_vec, 3);
    assert(it_single_simd == single_vec.end()); // 3 should be inserted at index 1

    // Test SIMD lower_bound with all elements equal
    std::vector<int> equal_vec = {2, 2, 2, 2};
    auto it_equal_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 2);
    assert(it_equal_simd == equal_vec.begin()); // 2 should be inserted at index 0

    it_equal_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 1);
    assert(it_equal_simd == equal_vec.begin()); // 1 should be inserted at index 0

    it_equal_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 3);
    assert(it_equal_simd == equal_vec.end()); // 3 should be inserted at index 4

    // Test SIMD lower_bound with default arguments
    auto it_default_simd = jrmwng::algorithm::simd::lower_bound(vec, 3);
    assert(it_default_simd == vec.begin() + 2); // 3 should be inserted at index 2

    auto it_d_default_simd = jrmwng::algorithm::simd::lower_bound(vec_d, 3.3);
    assert(it_d_default_simd == vec_d.begin() + 2); // 3.3 should be inserted at index 2

    auto it_custom_default_simd = jrmwng::algorithm::simd::lower_bound(custom_vec, new_value);
    assert(it_custom_default_simd == custom_vec.begin() + 2); // {4} should be inserted at index 2

    auto it_empty_default_simd = jrmwng::algorithm::simd::lower_bound(empty_vec, 1);
    assert(it_empty_default_simd == empty_vec.begin()); // 1 should be inserted at index 0

    auto it_single_default_simd = jrmwng::algorithm::simd::lower_bound(single_vec, 1);
    assert(it_single_default_simd == single_vec.begin()); // 1 should be inserted at index 0

    it_single_default_simd = jrmwng::algorithm::simd::lower_bound(single_vec, 3);
    assert(it_single_default_simd == single_vec.end()); // 3 should be inserted at index 1

    auto it_equal_default_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 2);
    assert(it_equal_default_simd == equal_vec.begin()); // 2 should be inserted at index 0

    it_equal_default_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 1);
    assert(it_equal_default_simd == equal_vec.begin()); // 1 should be inserted at index 0

    it_equal_default_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 3);
    assert(it_equal_default_simd == equal_vec.end()); // 3 should be inserted at index 4

    // Test SIMD lower_bound with floats
    std::vector<float> vec_f = {1.1f, 2.2f, 4.4f, 5.5f, 6.6f};
    auto it_f_simd = jrmwng::algorithm::simd::lower_bound(vec_f, 3.3f);
    assert(it_f_simd == vec_f.begin() + 2); // 3.3 should be inserted at index 2

    // Test SIMD lower_bound with custom predicate for floats
    struct CustomFloatType {
        float value;
        bool operator<(const CustomFloatType& other) const {
            return value < other.value;
        }
    };

    std::vector<CustomFloatType> custom_vec_f = {{1.1f}, {3.3f}, {5.5f}};
    CustomFloatType new_value_f = {4.4f};
    auto it_custom_f_simd = jrmwng::algorithm::simd::lower_bound(custom_vec_f, new_value_f, std::less<CustomFloatType>());
    assert(it_custom_f_simd == custom_vec_f.begin() + 2); // {4.4} should be inserted at index 2

    // Test SIMD lower_bound with empty vector of floats
    std::vector<float> empty_vec_f;
    auto it_empty_f_simd = jrmwng::algorithm::simd::lower_bound(empty_vec_f, 1.1f);
    assert(it_empty_f_simd == empty_vec_f.begin()); // 1.1 should be inserted at index 0

    // Test SIMD lower_bound with single element vector of floats
    std::vector<float> single_vec_f = {2.2f};
    auto it_single_f_simd = jrmwng::algorithm::simd::lower_bound(single_vec_f, 1.1f);
    assert(it_single_f_simd == single_vec_f.begin()); // 1.1 should be inserted at index 0

    it_single_f_simd = jrmwng::algorithm::simd::lower_bound(single_vec_f, 3.3f);
    assert(it_single_f_simd == single_vec_f.end()); // 3.3 should be inserted at index 1

    // Test SIMD lower_bound with all elements equal for floats
    std::vector<float> equal_vec_f = {2.2f, 2.2f, 2.2f, 2.2f};
    auto it_equal_f_simd = jrmwng::algorithm::simd::lower_bound(equal_vec_f, 2.2f);
    assert(it_equal_f_simd == equal_vec_f.begin()); // 2.2 should be inserted at index 0

    it_equal_f_simd = jrmwng::algorithm::simd::lower_bound(equal_vec_f, 1.1f);
    assert(it_equal_f_simd == equal_vec_f.begin()); // 1.1 should be inserted at index 0

    it_equal_f_simd = jrmwng::algorithm::simd::lower_bound(equal_vec_f, 3.3f);
    assert(it_equal_f_simd == equal_vec_f.end()); // 3.3 should be inserted at index 4

    // Test SIMD lower_bound with custom projection for integers
    struct CustomIntType {
        int value;
    };

    std::vector<CustomIntType> custom_vec_int = {{1}, {3}, {5}};
    auto it_custom_proj_int = jrmwng::algorithm::simd::lower_bound(custom_vec_int, 4, std::less<int>(), [](const CustomIntType& ct) { return ct.value; });
    assert(it_custom_proj_int == custom_vec_int.begin() + 2); // {4} should be inserted at index 2

    std::vector<CustomFloatType> custom_vec_float = {{1.1f}, {3.3f}, {5.5f}};
    auto it_custom_proj_float = jrmwng::algorithm::simd::lower_bound(custom_vec_float, 4.4f, std::less<float>(), [](const CustomFloatType& ct) { return ct.value; });
    assert(it_custom_proj_float == custom_vec_float.begin() + 2); // {4.4} should be inserted at index 2

    // Test SIMD lower_bound with custom projection for doubles
    struct CustomDoubleType {
        double value;
    };

    std::vector<CustomDoubleType> custom_vec_double = {{1.1}, {3.3}, {5.5}};
    auto it_custom_proj_double = jrmwng::algorithm::simd::lower_bound(custom_vec_double, 4.4, std::less<double>(), [](const CustomDoubleType& ct) { return ct.value; });
    assert(it_custom_proj_double == custom_vec_double.begin() + 2); // {4.4} should be inserted at index 2

    // Test SIMD lower_bound with SIMD projection for integers
    std::vector<int> vec_int = {1, 2, 4, 5, 6};
    auto it_simd_proj_int = jrmwng::algorithm::simd::lower_bound(vec_int, 3, std::less<int>(), [](const __m256i& v) { return v; });
    assert(it_simd_proj_int == vec_int.begin() + 2); // 3 should be inserted at index 2

    // Test SIMD lower_bound with SIMD projection for floats
    std::vector<float> vec_float = {1.1f, 2.2f, 4.4f, 5.5f, 6.6f};
    auto it_simd_proj_float = jrmwng::algorithm::simd::lower_bound(vec_float, 3.3f, std::less<float>(), [](const __m256& v) { return v; });
    assert(it_simd_proj_float == vec_float.begin() + 2); // 3.3 should be inserted at index 2

    // Test SIMD lower_bound with SIMD projection for doubles
    std::vector<double> vec_double = {1.1, 2.2, 4.4, 5.5, 6.6};
    auto it_simd_proj_double = jrmwng::algorithm::simd::lower_bound(vec_double, 3.3, std::less<double>(), [](const __m256d& v) { return v; });
    assert(it_simd_proj_double == vec_double.begin() + 2); // 3.3 should be inserted at index 2

    std::cout << "All SIMD tests passed!" << std::endl;
}

int main() {
    std::cout << "Compilation date and time: " << __DATE__ << " " << __TIME__ << std::endl;
    test_lower_bound_simd();
    return 0;
}
