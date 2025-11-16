using UnityEngine;

public static class EvolutionUtils
{

    public static NeuralNetworkData Clone(NeuralNetworkData src)
    {
        var dst = new NeuralNetworkData();
        dst.layers = new NeuralNetworkLayerData[src.layers.Length];

        for (int i = 0; i < src.layers.Length; i++)
        {
            var sl = src.layers[i];
            var dl = new NeuralNetworkLayerData
            {
                inputSize  = sl.inputSize,
                outputSize = sl.outputSize,
                activation = sl.activation,
                weights    = (float[])sl.weights.Clone(),
                biases     = (float[])sl.biases.Clone()
            };
            dst.layers[i] = dl;
        }

        return dst;
    }

    public static NeuralNetworkData Crossover(NeuralNetworkData a, NeuralNetworkData b, float mixProb = 0.5f)
    {
        // assume same shape; you can assert sizes here
        var child = Clone(a);

        for (int l = 0; l < child.layers.Length; l++)
        {
            var ca = a.layers[l];
            var cb = b.layers[l];
            var cc = child.layers[l];

            for (int i = 0; i < cc.weights.Length; i++)
            {
                cc.weights[i] = (Random.Range(0f,1f) < mixProb)
                    ? ca.weights[i]
                    : cb.weights[i];
            }

            for (int i = 0; i < cc.biases.Length; i++)
            {
                cc.biases[i] = (Random.Range(0f, 1f) < mixProb)
                    ? ca.biases[i]
                    : cb.biases[i];
            }
        }

        return child;
    }

    public static void Mutate(NeuralNetworkData data, float mutationRate, float mutationStdDev)
    {
        for (int l = 0; l < data.layers.Length; l++)
        {
            var layer = data.layers[l];

            for (int i = 0; i < layer.weights.Length; i++)
            {
                if (Random.Range(0f, 1f) < mutationRate)
                    layer.weights[i] += NextGaussian() * mutationStdDev;
            }

            for (int i = 0; i < layer.biases.Length; i++)
            {
                if (Random.Range(0f, 1f) < mutationRate)
                    layer.biases[i] += NextGaussian() * mutationStdDev;
            }
        }
    }

    private static float NextGaussian()
    {
        // Box–Muller
        float u1 = 1 - Random.Range(0f, 1f);
        float u2 = 1 - Random.Range(0f, 1f);
        return Mathf.Sqrt(-2 * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }
}
