# Lower Bound Test Project

This project implements a custom `lower_bound` function template that finds the first position in a sorted range where a given value could be inserted without violating the order. The implementation is provided in the [test_lower_bound.cpp](tests/test_lower_bound.cpp) file.

## Files Overview

- **include/lower_bound.hpp**: Contains the implementation of the `lower_bound` function template with detailed descriptions of its parameters and return type.
- **include/lower_bound_simd.hpp**: Contains SIMD-optimized implementations of the `lower_bound` function for different data types.
- **src/main.cpp**: The entry point for the test program, which includes the `lower_bound.hpp` header and tests the `lower_bound` function with various inputs.
- **tests/test_lower_bound.cpp**: Contains unit tests for the `lower_bound` function, validating its functionality with different data types and predicate functions.
- **tests/test_lower_bound_simd.cpp**: Contains unit tests for the SIMD-optimized `lower_bound` function.
- **CMakeLists.txt**: Configuration file for CMake, specifying the project name, C++ standard, include directories, and executable targets for the main program and tests.

## Building the Project

To build the project, follow these steps:

1. Ensure you have CMake installed on your system.
2. Open a terminal and navigate to the project directory.
3. Create a build directory:
   ```sh
   mkdir build
   cd build
   ```
4. Run CMake to configure the project:
   ```sh
   cmake ..
   ```
5. Build the project:
   ```sh
   cmake --build .
   ```

## Running the Tests

After building the project, you can run the tests using the following command:

```sh
./test_lower_bound
./test_lower_bound_simd
```

This will execute the unit tests defined in [test_lower_bound.cpp](tests/test_lower_bound.cpp) and [test_lower_bound_simd.cpp](tests/test_lower_bound_simd.cpp) and display the results in the terminal.

## Usage

The `lower_bound` function can be used to find the appropriate insertion point for a value in a sorted range. It takes a range defined by two iterators, the value to insert, and a predicate function that defines the comparison logic.

### Example Usage

#### Using Iterators with a Custom Predicate

```cpp
#include <vector>
#include <iostream>
#include "lower_bound.hpp"

int main() {
    std::vector<int> vec = {1, 2, 4, 5, 6};
    auto it = jrmwng::algorithm::lower_bound(vec.begin(), vec.end(), 3, std::less<int>());
    std::cout << "Insert position for 3: " << (it - vec.begin()) << std::endl; // Output: 2
    return 0;
}
```

#### Using Iterators with Default Predicate

```cpp
#include <vector>
#include <iostream>
#include "lower_bound.hpp"

int main() {
    std::vector<int> vec = {1, 2, 4, 5, 6};
    auto it = jrmwng::algorithm::lower_bound(vec.begin(), vec.end(), 3);
    std::cout << "Insert position for 3: " << (it - vec.begin()) << std::endl; // Output: 2
    return 0;
}
```

#### Using Ranges with Custom Comparison and Projection

```cpp
#include <vector>
#include <iostream>
#include "lower_bound.hpp"

int main() {
    std::vector<int> vec = {1, 2, 4, 5, 6};
    auto it = jrmwng::algorithm::ranges::lower_bound(vec, 3, std::less<int>(), [](int x) { return x; });
    std::cout << "Insert position for 3: " << (it - vec.begin()) << std::endl; // Output: 2
    return 0;
}
```

### SIMD-Optimized Usage

The SIMD-optimized `lower_bound` function can be used to find the appropriate insertion point for a value in a sorted range with improved performance for certain data types. It takes a range, the value to insert, and optional comparison and projection functions.

#### Using SIMD with Integers

```cpp
#include <vector>
#include <iostream>
#include "lower_bound_simd.hpp"

int main() {
    std::vector<int> vec = {1, 2, 4, 5, 6};
    auto it = jrmwng::algorithm::simd::lower_bound(vec, 3);
    std::cout << "Insert position for 3: " << (it - vec.begin()) << std::endl; // Output: 2
    return 0;
}
```

#### Using SIMD with Doubles

```cpp
#include <vector>
#include <iostream>
#include "lower_bound_simd.hpp"

int main() {
    std::vector<double> vec = {1.1, 2.2, 4.4, 5.5, 6.6};
    auto it = jrmwng::algorithm::simd::lower_bound(vec, 3.3);
    std::cout << "Insert position for 3.3: " << (it - vec.begin()) << std::endl; // Output: 2
    return 0;
}
```

#### Using SIMD with Custom Projection

```cpp
#include <vector>
#include <iostream>
#include "lower_bound_simd.hpp"

struct CustomType {
    int value;
};

int main() {
    std::vector<CustomType> vec = {{1}, {3}, {5}};
    CustomType value = {4};
    auto it = jrmwng::algorithm::simd::lower_bound(vec, value, std::less<int>(), [](const CustomType& ct) { return ct.value; });
    std::cout << "Insert position for 4: " << (it - vec.begin()) << std::endl; // Output: 2
    return 0;
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.