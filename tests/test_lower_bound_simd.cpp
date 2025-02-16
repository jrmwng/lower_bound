#include <gtest/gtest.h>
#include <vector>
#include "lower_bound_simd.hpp"

TEST(LowerBoundSimdTest, Integers) {
    std::vector<int> vec = {1, 2, 4, 5, 6};
    auto it_simd = jrmwng::algorithm::simd::lower_bound(vec, 3);
    EXPECT_EQ(it_simd, vec.begin() + 2);
}

TEST(LowerBoundSimdTest, Doubles) {
    std::vector<double> vec_d = {1.1, 2.2, 4.4, 5.5, 6.6};
    auto it_d_simd = jrmwng::algorithm::simd::lower_bound(vec_d, 3.3);
    EXPECT_EQ(it_d_simd, vec_d.begin() + 2);
}

TEST(LowerBoundSimdTest, CustomPredicate) {
    struct CustomType {
        int value;
        bool operator<(const CustomType& other) const {
            return value < other.value;
        }
    };

    std::vector<CustomType> custom_vec = {{1}, {3}, {5}};
    CustomType new_value = {4};
    auto it_custom_simd = jrmwng::algorithm::simd::lower_bound(custom_vec, new_value, std::less<CustomType>());
    EXPECT_EQ(it_custom_simd, custom_vec.begin() + 2);
}

TEST(LowerBoundSimdTest, EmptyVector) {
    std::vector<int> empty_vec;
    auto it_empty_simd = jrmwng::algorithm::simd::lower_bound(empty_vec, 1);
    EXPECT_EQ(it_empty_simd, empty_vec.begin());
}

TEST(LowerBoundSimdTest, SingleElementVector) {
    std::vector<int> single_vec = {2};
    auto it_single_simd = jrmwng::algorithm::simd::lower_bound(single_vec, 1);
    EXPECT_EQ(it_single_simd, single_vec.begin());

    it_single_simd = jrmwng::algorithm::simd::lower_bound(single_vec, 3);
    EXPECT_EQ(it_single_simd, single_vec.end());
}

TEST(LowerBoundSimdTest, AllElementsEqual) {
    std::vector<int> equal_vec = {2, 2, 2, 2};
    auto it_equal_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 2);
    EXPECT_EQ(it_equal_simd, equal_vec.begin());

    it_equal_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 1);
    EXPECT_EQ(it_equal_simd, equal_vec.begin());

    it_equal_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 3);
    EXPECT_EQ(it_equal_simd, equal_vec.end());
}

TEST(LowerBoundSimdTest, DefaultArguments) {
    std::vector<int> vec = {1, 2, 4, 5, 6};
    auto it_default_simd = jrmwng::algorithm::simd::lower_bound(vec, 3);
    EXPECT_EQ(it_default_simd, vec.begin() + 2);

    std::vector<double> vec_d = {1.1, 2.2, 4.4, 5.5, 6.6};
    auto it_d_default_simd = jrmwng::algorithm::simd::lower_bound(vec_d, 3.3);
    EXPECT_EQ(it_d_default_simd, vec_d.begin() + 2);

    struct CustomType {
        int value;
        bool operator<(const CustomType& other) const {
            return value < other.value;
        }
    };

    std::vector<CustomType> custom_vec = {{1}, {3}, {5}};
    CustomType new_value = {4};
    auto it_custom_default_simd = jrmwng::algorithm::simd::lower_bound(custom_vec, new_value);
    EXPECT_EQ(it_custom_default_simd, custom_vec.begin() + 2);

    std::vector<int> empty_vec;
    auto it_empty_default_simd = jrmwng::algorithm::simd::lower_bound(empty_vec, 1);
    EXPECT_EQ(it_empty_default_simd, empty_vec.begin());

    std::vector<int> single_vec = {2};
    auto it_single_default_simd = jrmwng::algorithm::simd::lower_bound(single_vec, 1);
    EXPECT_EQ(it_single_default_simd, single_vec.begin());

    it_single_default_simd = jrmwng::algorithm::simd::lower_bound(single_vec, 3);
    EXPECT_EQ(it_single_default_simd, single_vec.end());

    std::vector<int> equal_vec = {2, 2, 2, 2};
    auto it_equal_default_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 2);
    EXPECT_EQ(it_equal_default_simd, equal_vec.begin());

    it_equal_default_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 1);
    EXPECT_EQ(it_equal_default_simd, equal_vec.begin());

    it_equal_default_simd = jrmwng::algorithm::simd::lower_bound(equal_vec, 3);
    EXPECT_EQ(it_equal_default_simd, equal_vec.end());
}

TEST(LowerBoundSimdTest, Floats) {
    std::vector<float> vec_f = {1.1f, 2.2f, 4.4f, 5.5f, 6.6f};
    auto it_f_simd = jrmwng::algorithm::simd::lower_bound(vec_f, 3.3f);
    EXPECT_EQ(it_f_simd, vec_f.begin() + 2);
}

TEST(LowerBoundSimdTest, CustomPredicateFloats) {
    struct CustomFloatType {
        float value;
        bool operator<(const CustomFloatType& other) const {
            return value < other.value;
        }
    };

    std::vector<CustomFloatType> custom_vec_f = {{1.1f}, {3.3f}, {5.5f}};
    CustomFloatType new_value_f = {4.4f};
    auto it_custom_f_simd = jrmwng::algorithm::simd::lower_bound(custom_vec_f, new_value_f, std::less<CustomFloatType>());
    EXPECT_EQ(it_custom_f_simd, custom_vec_f.begin() + 2);
}

TEST(LowerBoundSimdTest, EmptyVectorFloats) {
    std::vector<float> empty_vec_f;
    auto it_empty_f_simd = jrmwng::algorithm::simd::lower_bound(empty_vec_f, 1.1f);
    EXPECT_EQ(it_empty_f_simd, empty_vec_f.begin());
}

TEST(LowerBoundSimdTest, SingleElementVectorFloats) {
    std::vector<float> single_vec_f = {2.2f};
    auto it_single_f_simd = jrmwng::algorithm::simd::lower_bound(single_vec_f, 1.1f);
    EXPECT_EQ(it_single_f_simd, single_vec_f.begin());

    it_single_f_simd = jrmwng::algorithm::simd::lower_bound(single_vec_f, 3.3f);
    EXPECT_EQ(it_single_f_simd, single_vec_f.end());
}

TEST(LowerBoundSimdTest, AllElementsEqualFloats) {
    std::vector<float> equal_vec_f = {2.2f, 2.2f, 2.2f, 2.2f};
    auto it_equal_f_simd = jrmwng::algorithm::simd::lower_bound(equal_vec_f, 2.2f);
    EXPECT_EQ(it_equal_f_simd, equal_vec_f.begin());

    it_equal_f_simd = jrmwng::algorithm::simd::lower_bound(equal_vec_f, 1.1f);
    EXPECT_EQ(it_equal_f_simd, equal_vec_f.begin());

    it_equal_f_simd = jrmwng::algorithm::simd::lower_bound(equal_vec_f, 3.3f);
    EXPECT_EQ(it_equal_f_simd, equal_vec_f.end());
}

TEST(LowerBoundSimdTest, CustomProjectionIntegers) {
    struct CustomIntType {
        int value;
    };

    std::vector<CustomIntType> custom_vec_int = {{1}, {3}, {5}};
    auto it_custom_proj_int = jrmwng::algorithm::simd::lower_bound(custom_vec_int, 4, std::less<int>(), [](const CustomIntType& ct) { return ct.value; });
    EXPECT_EQ(it_custom_proj_int, custom_vec_int.begin() + 2);
}

TEST(LowerBoundSimdTest, CustomProjectionFloats) {
    struct CustomFloatType {
        float value;
    };

    std::vector<CustomFloatType> custom_vec_float = {{1.1f}, {3.3f}, {5.5f}};
    auto it_custom_proj_float = jrmwng::algorithm::simd::lower_bound(custom_vec_float, 4.4f, std::less<float>(), [](const CustomFloatType& ct) { return ct.value; });
    EXPECT_EQ(it_custom_proj_float, custom_vec_float.begin() + 2);
}

TEST(LowerBoundSimdTest, CustomProjectionDoubles) {
    struct CustomDoubleType {
        double value;
    };

    std::vector<CustomDoubleType> custom_vec_double = {{1.1}, {3.3}, {5.5}};
    auto it_custom_proj_double = jrmwng::algorithm::simd::lower_bound(custom_vec_double, 4.4, std::less<double>(), [](const CustomDoubleType& ct) { return ct.value; });
    EXPECT_EQ(it_custom_proj_double, custom_vec_double.begin() + 2);
}

TEST(LowerBoundSimdTest, SimdProjectionIntegers) {
    std::vector<int> vec_int = {1, 2, 4, 5, 6};
    auto it_simd_proj_int = jrmwng::algorithm::simd::lower_bound(vec_int, 3, std::less<int>(), [](const __m256i& v) { return v; });
    EXPECT_EQ(it_simd_proj_int, vec_int.begin() + 2);
}

TEST(LowerBoundSimdTest, SimdProjectionFloats) {
    std::vector<float> vec_float = {1.1f, 2.2f, 4.4f, 5.5f, 6.6f};
    auto it_simd_proj_float = jrmwng::algorithm::simd::lower_bound(vec_float, 3.3f, std::less<float>(), [](const __m256& v) { return v; });
    EXPECT_EQ(it_simd_proj_float, vec_float.begin() + 2);
}

TEST(LowerBoundSimdTest, SimdProjectionDoubles) {
    std::vector<double> vec_double = {1.1, 2.2, 4.4, 5.5, 6.6};
    auto it_simd_proj_double = jrmwng::algorithm::simd::lower_bound(vec_double, 3.3, std::less<double>(), [](const __m256d& v) { return v; });
    EXPECT_EQ(it_simd_proj_double, vec_double.begin() + 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
