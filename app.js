"use strict";
//вход нейрона
class Input {
    constructor(neuron, weight) {
        this.neuron = neuron;
        this.weight = weight;
    }
}
//нейрон
class Neuron {
    constructor(layer, preLayer) {
        this._layer = layer;
        this.inputs = [];
        if (preLayer) {
            for (let i = 0; i < preLayer.neurons.length; i++) {
                const neuron = preLayer.neurons[i];
                const input = new Input(neuron, Math.random() - 0.5);
                this.inputs.push(input);
            }
        }
        else
            this.inputs.push(0);
    }
    get isFirst() {
        return !(this.inputs[0] instanceof Input);
    }
    get value() {
        if (!(this.inputs[0] instanceof Input))
            return this.inputs[0];
        else
            return this._layer._network.activationFunction(this.inputSum);
    }
    get inputSum() {
        let sum = 0;
        for (let i = 0; i < this.inputs.length; i++) {
            const input = this.inputs[i];
            if (input instanceof Input)
                sum += input.neuron.value * input.weight;
        }
        return sum;
    }
    set input(val) {
        if (!this.isFirst)
            return;
        this.inputs[0] = val;
    }
    set error(error) {
        if (this.isFirst)
            return;
        const wDelta = error * this._layer._network.derivativeFunction(this.inputSum);
        this.inputs.forEach((input) => {
            if (input instanceof Input) {
                input.weight -= input.neuron.value * wDelta * this._layer._network.learningRate;
                input.neuron.error = input.weight * wDelta;
            }
        });
    }
}
//слой
class Layer {
    constructor(neuronsCount, preLayer, network) {
        this._network = network;
        this.neurons = [];
        for (let i = 0; i < neuronsCount; i++) {
            this.neurons.push(new Neuron(this, preLayer));
        }
    }
    get isFirst() {
        return this.neurons[0].isFirst;
    }
    set input(values) {
        if (!this.isFirst)
            return;
        if (!Array.isArray(values))
            return;
        if (values.length !== this.neurons.length)
            return;
        values.forEach((value, i) => this.neurons[i].input = value);
    }
}
//сеть
class Network {
    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    static sigmoidDerivative(x) {
        return Network.sigmoid(x) * (1 - Network.sigmoid(x));
    }
    constructor(inputSize, outputSize, hiddenLayersCount = 1, learningRate = 0.5) {
        this.activationFunction = Network.sigmoid;
        this.derivativeFunction = Network.sigmoidDerivative;
        this.learningRate = learningRate;
        this.layers = [new Layer(inputSize, null, this)];
        for (let i = 0; i < hiddenLayersCount; i++) {
            /*console.log(`${inputSize * 2 - 1}-one`)
            console.log(`${Math.ceil((inputSize * 2 / 3) + outputSize)}-ceil`)*/
            const layerSize = Math.min(inputSize * 2 - 1, Math.ceil((inputSize * 2 / 3) + outputSize));
            this.layers.push(new Layer(layerSize, this.layers[this.layers.length - 1], this));
        }
        this.layers.push(new Layer(outputSize, this.layers[this.layers.length - 1], this));
    }
    set input(values) {
        this.layers[0].input = values;
    }
    get prediction() {
        const prediction = [];
        const outputLayer = this.layers[this.layers.length - 1];
        for (let i = 0; i < outputLayer.neurons.length; i++) {
            prediction.push(outputLayer.neurons[i].value);
        }
        return prediction;
    }
    trainOnce(dataSet) {
        console.log("training");
        if (!Array.isArray(dataSet))
            return;
        dataSet.forEach((data) => {
            const [input, expected] = data;
            this.input = input;
            this.prediction.forEach((r, i) => {
                this.layers[this.layers.length - 1].neurons[i].error = r - expected[i];
            });
        });
    }
    train(dataSet, epochs = 5000) {
        for (let i = 0; i < epochs; i++)
            this.trainOnce(dataSet);
    }
}
/*
console.log("--------debug-network--------")
const network: Network = new Network(2, 1);

const data: number[][][] = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
    ];
  network.trainOnce(data)

  const testData: number[][] = [
    [0, 0]
  ];
  testData.forEach((input) => {
    network.input = input;
    console.log(`${input[0]} XoR ${input[1]} => ${network.prediction}`)
  });

*/
/*
console.log("--------network1--------")
const network: Network = new Network(2, 1);

const data: number[][][] = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
    ];
  network.train(data)

  const testData: number[][] = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ];
  testData.forEach((input) => {
    network.input = input;
    console.log(`${input[0]} XoR ${input[1]} => ${network.prediction}`)
  });




console.log("--------network2--------")
const network2: Network = new Network(3, 1, 2)
const data2: number[][][]= [
    [[0, 0, 0], [0]],
    [[0, 0, 1], [1]],
    [[0, 1, 0], [0]],
    [[0, 1, 1], [1]],
    [[1, 0, 0], [0]],
    [[1, 0, 1], [1]],
    ];
  network2.train(data2)

  const testData2: number[][] = [
    [1, 1, 0],
    [1, 1, 1]
  ]
  testData2.forEach((input) => {
    network2.input = input;
    console.log(`${input} is last one? => ${network2.prediction}`)
  })
  
  
  network2.layers.forEach((layer, i) =>{
        console.log(`-----------layer${i+1}--------`)
        layer.neurons.forEach((neuron, i)=>{
            console.log(`-----------neuron${i+1}--------`)
            console.log(neuron.inputs)
        })
  })*/
console.log("--------network3--------");
const network2 = new Network(15, 15, 5);
const data2 = [
    [[1, 1, 1,
            1, 0, 1,
            1, 0, 1,
            1, 0, 1,
            1, 1, 1,],
        [1, 0, 0, 0]],
    [[0, 1, 0,
            1, 1, 0,
            0, 1, 0,
            0, 1, 0,
            1, 1, 1,],
        [0, 1, 0, 0]],
    [[0, 1, 0,
            1, 0, 1,
            0, 0, 1,
            0, 1, 0,
            1, 1, 1,],
        [0, 0, 1, 0]],
    [[1, 1, 1,
            0, 0, 1,
            1, 1, 1,
            0, 0, 1,
            1, 1, 1,],
        [0, 0, 0, 1]]
];
const testData2 = [
    [1, 1, 1,
        1, 0, 1,
        1, 0, 1,
        1, 0, 1,
        1, 1, 1,],
    [0, 1, 0,
        1, 1, 0,
        0, 1, 0,
        0, 1, 0,
        1, 1, 1,],
    [0, 1, 0,
        1, 0, 1,
        0, 0, 1,
        0, 1, 0,
        1, 1, 1,],
    [1, 1, 1,
        0, 0, 1,
        1, 1, 1,
        0, 0, 1,
        1, 1, 1,],
];
console.log("------------0--------------");
testData2.forEach((input) => {
    network2.input = input;
    console.log(`${input} is => ${network2.prediction}`);
});
console.log("------------100------------");
network2.train(data2, 100);
testData2.forEach((input) => {
    network2.input = input;
    console.log(`${input} is => ${network2.prediction}`);
});
console.log("------------1000------------");
network2.train(data2, 1000);
testData2.forEach((input) => {
    network2.input = input;
    console.log(`${input} is => ${network2.prediction}`);
});
console.log("------------100000------------");
network2.train(data2, 10000);
testData2.forEach((input) => {
    network2.input = input;
    console.log(`${input} is => ${network2.prediction}`);
});
