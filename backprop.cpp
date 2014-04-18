#include <cmath>
#include <iostream>

// Update the weights through back propagation
void backPropagate(vector<double>* outputs, int parent) {
  Layer *outputLayer = (*layers)[hiddenLayersCount + 1];
  for (int i = 0; i < outputs->size(); i++) {
    Node *n = outputLayer->getNode(i);
    double adjusted = -n->getValue();
    if (i == parent) {
      adjusted += 1;
    }
    n->setDelta(sigmoidPrime(n->getActivation()) * adjusted);
  }
  // Go backwards from output to input propagating deltas
  for (int l = hiddenLayersCount; l >= 0; l--) {
    Layer *curr = (*layers)[l], *downstream = (*layers)[l+1];

    for (int i = 0; i < curr->NodeCount(); i++) {
      double sum = 0;
      Node *n = curr->getNode(i);
      for (int j = 0; j < downstream->NodeCount(); j++) {
        sum += downstream->getNode(j)->getWeight(i)
            * downstream->getNode(j)->getDelta();
      }
      n->setDelta(sigmoidPrime(n->getActivation()) * sum);
      for (int j = 0; j < downstream->NodeCount(); j++) {
        downstream->getNode(j)->updateWeight(i,
            learnRate * sigmoid(n->getActivation())
            * downstream->getNode(j)->getDelta());
      }
    }
  }
}

// Computing sigmoid
inline double sigmoid(double activation) {
  return 1.0 / (1.0 + exp(-activation / respThreshold));
}

// Derivitive of sigmoid
inline double sigmoidPrime(double activation) {
  double exponential = exp(activation / respThreshold);
  return exponential / (respThreshold * pow(exponential + 1, 2));
}
