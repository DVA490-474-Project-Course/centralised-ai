//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-10-01 by Jacob Johansson
// Description: Headers for mlp.cc.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef MLP_H
#define MLP_H

#include <vector>

namespace centralised_ai{
namespace collective_robot_behaviour{

// Declaration of ReLU for a single value
double relu(double x);

// Declaration of ReLU for a vector of values
std::vector<double> relu(const std::vector<double>& x);

} // namespace centralised_ai
} // namespace collective_robot_behaviour

#endif // MLP_H
