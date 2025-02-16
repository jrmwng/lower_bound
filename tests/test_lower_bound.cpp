#include <gtest/gtest.h>
#include <vector>
#include "lower_bound.hpp"

TEST(LowerBoundTest, Integers) {
    std::vector<int> vec = {1, 2, 4, 5, 6};
    std::vector<int> test_values = {3, 0, 7};
    std::vector<size_t> expected_indices = {2, 0, 5};
    for (size_t i = 0; i < test_values.size(); ++i) {
        auto it = jrmwng::algorithm::lower_bound(vec.begin(), vec.end(), test_values[i], std::less<int>());
        EXPECT_EQ(it, vec.begin() + expected_indices[i]);
    }
}

TEST(LowerBoundTest, Doubles) {
    std::vector<double> vec_d = {1.1, 2.2, 4.4, 5.5, 6.6};
    std::vector<double> test_values_d = {3.3, 0.0, 7.7};
    std::vector<size_t> expected_indices_d = {2, 0, 5};
    for (size_t i = 0; i < test_values_d.size(); ++i) {
        auto it_d = jrmwng::algorithm::lower_bound(vec_d.begin(), vec_d.end(), test_values_d[i], std::less<double>());
        EXPECT_EQ(it_d, vec_d.begin() + expected_indices_d[i]);
    }
}

TEST(LowerBoundTest, CustomPredicate) {
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
        EXPECT_EQ(it_custom, custom_vec.begin() + expected_indices_custom[i]);
    }
}

TEST(LowerBoundTest, EmptyVector) {
    std::vector<int> empty_vec;
    auto it_empty = jrmwng::algorithm::lower_bound(empty_vec.begin(), empty_vec.end(), 1);
    EXPECT_EQ(it_empty, empty_vec.begin());
}

TEST(LowerBoundTest, SingleElementVector) {
    std::vector<int> single_vec = {2};
    std::vector<int> test_values_single = {1, 3};
    std::vector<size_t> expected_indices_single = {0, 1};
    for (size_t i = 0; i < test_values_single.size(); ++i) {
        auto it_single = jrmwng::algorithm::lower_bound(single_vec.begin(), single_vec.end(), test_values_single[i]);
        EXPECT_EQ(it_single, single_vec.begin() + expected_indices_single[i]);
    }
}

TEST(LowerBoundTest, AllElementsEqual) {
    std::vector<int> equal_vec = {2, 2, 2, 2};
    std::vector<int> test_values_equal = {2, 1, 3};
    std::vector<size_t> expected_indices_equal = {0, 0, 4};
    for (size_t i = 0; i < test_values_equal.size(); ++i) {
        auto it_equal = jrmwng::algorithm::lower_bound(equal_vec.begin(), equal_vec.end(), test_values_equal[i]);
        EXPECT_EQ(it_equal, equal_vec.begin() + expected_indices_equal[i]);
    }
}

TEST(LowerBoundTest, RangeBasedIntegers) {
    std::vector<int> vec = {1, 2, 4, 5, 6};
    std::vector<int> test_values = {3, 0, 7};
    std::vector<size_t> expected_indices = {2, 0, 5};
    for (size_t i = 0; i < test_values.size(); ++i) {
        auto it_range = jrmwng::algorithm::ranges::lower_bound(vec, test_values[i], std::less<int>(), [](int i) { return i; });
        EXPECT_EQ(it_range, vec.begin() + expected_indices[i]);
    }
}

TEST(LowerBoundTest, RangeBasedDoubles) {
    std::vector<double> vec_d = {1.1, 2.2, 4.4, 5.5, 6.6};
    std::vector<double> test_values_d = {3.3, 0.0, 7.7};
    std::vector<size_t> expected_indices_d = {2, 0, 5};
    for (size_t i = 0; i < test_values_d.size(); ++i) {
        auto it_d_range = jrmwng::algorithm::ranges::lower_bound(vec_d, test_values_d[i], std::less<double>(), [](double d) { return d; });
        EXPECT_EQ(it_d_range, vec_d.begin() + expected_indices_d[i]);
    }
}

TEST(LowerBoundTest, RangeBasedCustomPredicate) {
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
        auto it_custom_range = jrmwng::algorithm::ranges::lower_bound(custom_vec, test_values_custom[i], std::less<CustomType>(), [](const CustomType& ct) { return ct; });
        EXPECT_EQ(it_custom_range, custom_vec.begin() + expected_indices_custom[i]);
    }
}

TEST(LowerBoundTest, RangeBasedEmptyVector) {
    std::vector<int> empty_vec;
    auto it_empty_range = jrmwng::algorithm::ranges::lower_bound(empty_vec, 1, std::less<int>(), [](int i) { return i; });
    EXPECT_EQ(it_empty_range, empty_vec.begin());
}

TEST(LowerBoundTest, RangeBasedSingleElementVector) {
    std::vector<int> single_vec = {2};
    std::vector<int> test_values_single = {1, 3};
    std::vector<size_t> expected_indices_single = {0, 1};
    for (size_t i = 0; i < test_values_single.size(); ++i) {
        auto it_single_range = jrmwng::algorithm::ranges::lower_bound(single_vec, test_values_single[i], std::less<int>(), [](int i) { return i; });
        EXPECT_EQ(it_single_range, single_vec.begin() + expected_indices_single[i]);
    }
}

TEST(LowerBoundTest, RangeBasedAllElementsEqual) {
    std::vector<int> equal_vec = {2, 2, 2, 2};
    std::vector<int> test_values_equal = {2, 1, 3};
    std::vector<size_t> expected_indices_equal = {0, 0, 4};
    for (size_t i = 0; i < test_values_equal.size(); ++i) {
        auto it_equal_range = jrmwng::algorithm::ranges::lower_bound(equal_vec, test_values_equal[i], std::less<int>(), [](int i) { return i; });
        EXPECT_EQ(it_equal_range, equal_vec.begin() + expected_indices_equal[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}