type Functor = fn(signal: f32) -> f32;

fn standard_fun(signal: f32) -> f32 {
    1.0 / (1.0 + (-signal).exp())
}

struct Neuron {
    weights: Vec<f32>,
    weight_sum: f32,
    functor: Functor,
}

struct Layer {
    neurons: Vec<Neuron>,
}

#[derive(Clone, Debug)]
pub struct Configuration {
    num_inputs: usize,
    num_layers: usize,
    layer_size: usize,
    num_outputs: usize,
}

pub struct Network {
    layers: Vec<Layer>,
    temp: Vec<f32>,
}

impl Network {
    pub fn new(config: &Configuration) -> Self {
        Self {
            layers: (0..=config.num_layers)
                .map(|li| {
                    let num_neurons = if li == config.num_layers {
                        config.num_outputs
                    } else {
                        config.layer_size
                    };
                    let num_weights = if li == 0 {
                        config.num_inputs
                    } else {
                        config.layer_size
                    };
                    Layer {
                        neurons: (0..num_neurons)
                            .map(|_| Neuron {
                                weights: vec![1.0; num_weights],
                                weight_sum: num_weights as f32,
                                functor: standard_fun,
                            })
                            .collect(),
                    }
                })
                .collect(),
            temp: Vec::new(),
        }
    }

    pub fn infer(&mut self, inputs: &[f32]) -> &[f32] {
        self.temp.clear();
        self.temp.clone_from_slice(inputs);
        for layer in self.layers.iter() {
            let input_count = self.temp.len();
            for neuron in layer.neurons.iter() {
                assert_eq!(neuron.weights.len(), input_count);
                let mut sum = 0.0;
                for (&signal, &weight) in self.temp.iter().zip(neuron.weights.iter()) {
                    sum += signal * weight;
                }
                self.temp.push((neuron.functor)(sum / neuron.weight_sum));
            }
            self.temp.drain(..input_count);
        }
        &self.temp
    }
}
