package dl.layers;

import java.util.Arrays;
import dl.loss.*;
import dl.optimizer.*;
import dl.utils.*;
// import dl.layers.*;

public class Layer {
	private Vector input;
	private Vector output;
	private Vector predictions;
	private Matrix weights;
	private Matrix gradients;

	public Layer(Vector input, Vector output){
		this.input = input;
		this.output = output;

		// initialize the weight matrix
		this.weights = new Matrix(this.output.getElements().length, this.input.getElements().length);

		// initialize the preidctions vector
		double[] prediction_elems = new double[this.output.getElements().length];
		for (int i = 0; i < this.output.getElements().length; i++){
			prediction_elems[i] = this.input.dot(this.weights.getVector(i));
		}

		this.predictions = new Vector(prediction_elems, false);
	}

	public void setInput(Vector input){
		this.input = input;
	}
	public Vector getInput(){
		return this.input;
	}

	public void setOutput(Vector output){
		this.output = output;
	}
	public Vector getOutput(){
		return this.output;
	}

	public void setPrediction(Vector predictions){
		this.predictions = predictions;
	}
	public Vector getPrediction(){
		return this.predictions;
	}

	public void setWeights(Matrix weights){
		this.weights = weights;
	}
	public Matrix getWeights(){
		return this.weights;
	}

	public void setGradients(Matrix gradients){
		this.gradients = gradients;
	}
	public Matrix getGradients(){
		return this.gradients;
	}
}