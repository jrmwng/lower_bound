#include <iostream>
#include <vector>
#include "lower_bound.hpp"

int main() {
    // Sample sorted vector
    std::vector<int> vec = {1, 2, 4, 5, 6};

    // Test lower_bound with a value that exists
    int value1 = 4;
    auto it1 = jrmwng::algorithm::lower_bound(vec.begin(), vec.end(), value1, std::less<int>());
    std::cout << "Lower bound for " << value1 << " is at index: " << std::distance(vec.begin(), it1) << std::endl;

    // Test lower_bound with a value that does not exist
    int value2 = 3;
    auto it2 = jrmwng::algorithm::lower_bound(vec.begin(), vec.end(), value2, std::less<int>());
    std::cout << "Lower bound for " << value2 << " is at index: " << std::distance(vec.begin(), it2) << std::endl;

    // Test lower_bound with a value greater than all elements
    int value3 = 7;
    auto it3 = jrmwng::algorithm::lower_bound(vec.begin(), vec.end(), value3, std::less<int>());
    std::cout << "Lower bound for " << value3 << " is at index: " << std::distance(vec.begin(), it3) << std::endl;

    return 0;
}