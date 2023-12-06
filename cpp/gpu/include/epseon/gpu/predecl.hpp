#pragma once

namespace epseon::gpu {
    namespace cpp {

        template <typename FP>
        class Algorithm;

        template <typename FP>
        class VibwaAlgorithm;

        template <typename FP>
        class TaskHandle;

        template <typename FP>
        struct HardwareConfig;

        template <typename FP>
        class PotentialSource;

        template <typename FP>
        class PotentialFileLoader;

        template <typename FP>
        class MorsePotentialConfig;

        template <typename FP>
        class MorsePotentialGenerator;

        template <typename FP>
        struct ShaderBuffersRequirements;

        template <typename FP>
        class AlgorithmConfig;

        template <typename FP>
        class VibwaAlgorithmConfig;

        template <typename FP>
        class TaskConfigurator;

        class ComputeDeviceInterface;

    } // namespace cpp

    namespace python {
        template <typename FP>
        class TaskHandle;

        class ComputeDeviceInterface;

        class MorsePotentialConfig;

        template <typename FP>
        class TaskConfigurator;

        class EpseonComputeContext;
    } // namespace python
} // namespace epseon::gpu
