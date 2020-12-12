declare module "nn.js" {
  type PredictFunction = (inputs: number[]) => number[];
  type TrainFunction = (inputs: number[], outputs: number[]) => number;
  interface NN extends Array<number> {
    predict: PredictFunction;
    train: TrainFunction;
  }
  interface Layer extends Array<number> {
    forward: (x: number[]) => number[];
    backward: (x: number[], err: number[], rate: number) => number[];
  }
  interface ActivationFunction {
    f: (x: number) => number;
    df: (x: number) => number;
  }
  export const NN: (...layers: Layer[]) => NN;
  export const Dense: ({
    inputs: number;
    units: number;
    act?: ActivationFunction;
    bias?: boolean;
    weights?: number[];
  });
  export const sigmoid: ActivationFunction;
  export const relu: ActivationFunction;
  export const lrelu: ActivationFunction;
  export const linear: ActivationFunction;
  export const softplus: ActivationFunction;
}
