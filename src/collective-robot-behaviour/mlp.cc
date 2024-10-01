//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-10-01 by Jacob Johansson
// Description: Stores all functions that are related to Multi-layer Perceptron networks.
// License: See LICENSE file for license details.
//==============================================================================

#include "mlp.h"
#include <algorithm> // for std::max, etc

// Implementation of ReLU for a single value
double relu(double x){
	return std::max(0.0, x);
}

// Implementation of ReLU for a vector of values
std::vector<double> relu(const std::vector<double>& x){
	std::vector<double> result;

	for (double value:x){
		result.push_back(std::max(0.0, value));
	}

	return result;
}