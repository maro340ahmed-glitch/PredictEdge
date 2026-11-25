
#pragma once

#include <math.h>

namespace PredictEdgeML {

static constexpr int FEATURES = 6;

static constexpr float ALERT_THRESHOLD = 0.80f;
static constexpr float RESET_THRESHOLD = 0.60f;

static constexpr unsigned long SUMMARY_PERIOD_SECONDS = 8UL;

// Original training metadata
static constexpr unsigned long TREND_WINDOW_SECONDS = 40UL;
static constexpr unsigned long INFERENCE_PERIOD_SECONDS = 300UL;
static constexpr unsigned long FAILURE_HORIZON_SECONDS = 3600UL;

static const float MEAN[FEATURES] = {
  36.03881532f,
  5.367306788f,
  0.2366282635f,
  0.0005256277195f,
  8.116619055e-05f,
  1.652824629e-05f
};

static const float SCALE[FEATURES] = {
  6.226381395f,
  1.224725333f,
  0.1828603837f,
  0.001253755984f,
  0.0002739968144f,
  5.479675668e-05f
};

static const float WEIGHT[FEATURES] = {
  1.691262953f,
  1.40638433f,
  1.728686754f,
  -0.2322045421f,
  0.0009204517643f,
  -0.3839193813f
};

static const float BIAS = -1.503511772f;

inline float predictRisk(const float *x) {
  float z = BIAS;

  for (int i = 0; i < FEATURES; ++i) {
    float normalized =
      (x[i] - MEAN[i]) / SCALE[i];

    z += WEIGHT[i] * normalized;
  }

  z = fmaxf(-35.0f, fminf(35.0f, z));

  return 1.0f / (1.0f + expf(-z));
}

}
